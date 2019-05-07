from torchtext import data
from torchtext.vocab import Vectors, GloVe
# from nltk import word_tokenize
import os
import torch

CURRENT_PATH = os.path.dirname(__file__)

# VERSION_DATA = 'v3'
VERSION_DATA = 'v4'
TRAIN_FN = f'converted-{VERSION_DATA}_train_petfinder_data_study2.csv'
VAL_FN = f'converted-{VERSION_DATA}_val_petfinder_data_study2.csv'
TEST_FN = f'converted-{VERSION_DATA}_test_petfinder_data_study2.csv'


def sent_tokenize(x):
    return x.split('<end>')


def tokenize(x):
    return x.split()


# TODO make for test, only load test file.
def load_data(train_bsize=32, bsize=64, embedding_length=200):
    document_field_file = f'{CURRENT_PATH}/models/document-{VERSION_DATA}.glove-{embedding_length}.pt'
    label_field_file = f'{CURRENT_PATH}/models/label-{VERSION_DATA}.glove-{embedding_length}.pt'

    if not os.path.exists(document_field_file):
        sentence = data.Field(tokenize=tokenize, lower=True, batch_first=True)
        document = data.NestedField(nesting_field=sentence, tokenize=sent_tokenize, include_lengths=True)
        label = data.LabelField()
    else:
        document = torch.load(document_field_file)
        label = torch.load(label_field_file)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='datasets',
        train=TRAIN_FN,
        validation=VAL_FN,
        test=TEST_FN,
        format='tsv',
        fields=[('id', None), ('document', document), ('label', label)]
    )

    if not os.path.exists(document_field_file):
        document.build_vocab(train_data, val_data, test_data, vectors=GloVe(name='6B', dim=embedding_length))
        label.build_vocab(train_data, val_data, test_data)

        print(f'Saving document field to {document_field_file}')
        torch.save(document, document_field_file)
        torch.save(label, label_field_file)

    word_embeddings = document.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(document.vocab)))
    print("Vector size of Text Vocabulary: ", document.vocab.vectors.size())
    print("Label Length: " + str(len(label.vocab)))

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_sizes=(train_bsize, bsize, bsize),
        sort_key=lambda x: len(x.document),
        device='cpu:0',
        repeat=False,
        sort_within_batch=True
        # repeat=True
    )

    vocab_size = len(document.vocab)

    return document, label, vocab_size, word_embeddings, train_iter, valid_iter, test_iter


if __name__ == '__main__':
    # prepare data:
    DOCUMENT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data(train_bsize=2,
                                                                                                bsize=2 * 2,
                                                                                                embedding_length=200)

    print('vocab_size', vocab_size)

    for idx, batch in enumerate(train_iter):
        document = batch.document
        target = batch.label
        print('document', document)
        print('target', target)
        break

    print(DOCUMENT.vocab.itos[1])
    print(DOCUMENT.vocab.itos[0])

    out_document = []
    for sen in document[0][0]:
        sen = sen.tolist()
        out_sen = []
        for w in sen:
            out_w = DOCUMENT.vocab.itos[w]
            if out_w != '<pad>':
                out_sen.append(out_w)
        out_sen = ' '.join(out_sen)
        out_document.append(out_sen)

    print('<end>'.join(out_document))
