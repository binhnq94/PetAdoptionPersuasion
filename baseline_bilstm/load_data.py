from torchtext import data
from torchtext.vocab import Vectors, GloVe
# from nltk import word_tokenize
import os
import torch


DATA_VERSION = 'v2'
DATA_DIR = 'datasets/190524'


def tokenize(x):
    return x.split()


# TODO make for test, only load test file.
def load_data(train_bsize=32, bsize=64, embedding_length=200):
    text_field_file = f'baseline_bilstm/models/TEXT-databalance-{DATA_VERSION}.glove-{embedding_length}.pt'
    label_field_file = f'baseline_bilstm/models/LABEL-databalance-{DATA_VERSION}.glove-{embedding_length}.pt'

    if not os.path.exists(text_field_file):
        TEXT = data.Field(tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
        LABEL = data.LabelField()
    else:
        TEXT = torch.load(text_field_file)
        LABEL = torch.load(label_field_file)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=DATA_DIR,
        # train='converted-v2_train_petfinder_data_study2.csv',
        # validation='converted-v2_val_petfinder_data_study2.csv',
        # test='converted-v2_test_petfinder_data_study2.csv',
        train=f'converted-{DATA_VERSION}_train.csv',
        validation=f'converted-{DATA_VERSION}_val.csv',
        test=f'converted-{DATA_VERSION}_test.csv',
        format='tsv',
        fields=[('id', None), ('text', TEXT), ('label', LABEL)]
    )

    if not os.path.exists(text_field_file):
        TEXT.build_vocab(train_data, val_data, test_data, vectors=GloVe(name='6B', dim=embedding_length))
        LABEL.build_vocab(train_data, val_data, test_data)

        print(f'Saving TEXT field to {text_field_file}')
        torch.save(TEXT, text_field_file)
        torch.save(LABEL, label_field_file)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))
    print("Label Vocab: " + str(LABEL.vocab))

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_sizes=(train_bsize, bsize, bsize),
        sort_key=lambda x: len(x.text),
        device='cpu:0',
        repeat=False,
        sort_within_batch=True
        # repeat=True
    )

    vocab_size = len(TEXT.vocab)

    return TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter


if __name__ == '__main__':
    # prepare data:
    TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data(train_bsize=2,
                                                                                            bsize=2,
                                                                                            embedding_length=200)

    print('vocab_size', vocab_size)

    for idx, batch in enumerate(valid_iter):
        text = batch.text
        target = batch.label
        print(text)
        print(target)
        break

    print(TEXT.vocab.itos[1])
    print(TEXT.vocab.itos[0])

    print(TEXT.vocab.vectors[1])
    print(TEXT.vocab.vectors[0])

