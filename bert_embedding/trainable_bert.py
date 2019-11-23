from flair.embeddings import BertEmbeddings, Sentence, ScalarMix
import pandas as pd
from typing import List
import flair
import torch


def read_all_data(list_files=None):
    version_data = 'v6'
    train_fn = f'datasets/190524/converted-{version_data}_train.csv'
    val_fn = f'datasets/190524/converted-{version_data}_val.csv'
    test_fn = f'datasets/190524/converted-{version_data}_test.csv'
    data_files = [train_fn, val_fn, test_fn]
    list_files = data_files if list_files is None else list_files
    out_dict = {}
    for fn in list_files:
        df = pd.read_csv(fn, sep='\t', names=['id', 'document', 'label'], header=None)
        for row in df.itertuples(index=False):
            id_ = int(row.id)
            assert id_ not in out_dict
            out_dict[id_] = row.document
    return out_dict


ID2DOCUMENT = read_all_data()


class BertEmbeddingsTrainable(BertEmbeddings):
    def __init__(self, *args, trainable=False, **kwargs):
        super(BertEmbeddingsTrainable, self).__init__(*args, **kwargs)
        self.trainable = True

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(flair.device)
        if not self.trainable:
            self.model.eval()
        _, _, all_encoder_layers = self.model(
            all_input_ids, token_type_ids=None, attention_mask=all_input_masks
        )

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        if self.use_scalar_mix:
                            layer_output = all_encoder_layers[int(layer_index)][
                                sentence_index
                            ]
                        else:
                            layer_output = (
                                all_encoder_layers[int(layer_index)]
                                .detach()
                                .cpu()[sentence_index]
                            )
                        all_layers.append(layer_output[token_index])

                    if self.use_scalar_mix:
                        sm = ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences
