from typing import *
import re
import string
import numpy as np
from .create_w2v import train_path
import os
from .file_utils import CURRENT_DIR, DATA_DIR

Data = List[Tuple[List[str], str]]

RE_D = re.compile(r'\d')


def has_digit(word):
    # return any(char.isdigit() for char in word)
    return bool(RE_D.search(word))


def clean_words(words):
    result = []
    for word in words:
        if word in string.punctuation:
            continue
        if has_digit(word):
            continue
        result.append(word)
    return result


def read_data(fpath: str, tokenize) -> Data:
    data = []
    with open(fpath, encoding='utf-8') as fi:
        for line in fi:
            line = line.strip('\n')
            if line:
                _id, sentence, label = line.split('\t')
                assert label in ['Adopted', 'Unadopted']
                sentence = sentence.lower()
                words = tokenize(sentence)
                words = clean_words(words)

                data.append((words, label))

    return data


class TfIdf:

    def __init__(self):
        tokenize = lambda x: x.split()
        train_data = read_data(os.path.join(DATA_DIR, train_path), tokenize=tokenize)
        self.document_count = len(train_data)
        self.vocab, df = self.create_vocab(train_data)  # type: list, dict
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.df = [df[w] for w in self.vocab]
        del train_data

    @staticmethod
    def create_vocab(data):
        df = TfIdf.compute_df(data)

        len_data = len(data)
        print(len_data)

        vocab = list(df.keys())
        print('before', len(vocab))

        count_reject = {
            '<': 0,
            '>': 0
        }

        for word in vocab:
            # if word in string.punctuation:
            #     del df[word]
            #     count_reject['punc'] += 1
            # elif df[word] < int(len_data/20):
            # elif df[word] < int(len_data/30):
            if df[word] <= 5:
                # if df[word] <= 10:
                # if df[word] <= 100:
                del df[word]
                count_reject['<'] += 1
            # elif df[word] > int(len_data/10):
            # elif df[word] > 100000:
            elif df[word] > 50000:
                del df[word]
                count_reject['>'] += 1

        print(count_reject)
        new_vocab = list(df.keys())

        print('after', len(new_vocab))

        with open(os.path.join(CURRENT_DIR, 'resources/vocab.txt'), 'w', encoding='utf-8') as fo:
            for word in new_vocab:
                fo.write(f'{word}\n')

        with open(os.path.join(CURRENT_DIR, 'resources/df.txt'), 'w', encoding='utf-8') as fo:
            for word in df:
                fo.write(f'{word}\t{df[word]}\n')

        return new_vocab, df

    @staticmethod
    def compute_df(data: Data):
        df = {}
        for document, label in data:
            set_document = set(document)
            for word in set_document:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1
        return df

    def words_to_indices(self, words):
        sequence_indices = []

        for word in words:
            # if word in self.vocab:
            #     sequence_indices.append(self.vocab.index(word))
            sequence_indices.append(self.stoi.get(word, -1))

        return [idx for idx in sequence_indices if idx >= 0]

    def preprocess_sen(self, sequence, tokenize=None):
        if tokenize is None:
            tokenize = lambda x: x.split()
        # tokenize = tokenize or lambda x: x.split()

        sequence = sequence.lower()
        words = tokenize(sequence)
        words = clean_words(words)
        words = self.words_to_indices(words)
        return words

    def sequence_to_vector(self, sequence, tokenize=None):
        words = self.preprocess_sen(sequence, tokenize)
        return self.words_to_vector(words)

    def words_to_vector(self, words: List[int], sparse=False, mode='tfidf'):
        """
        :param mode:
        :param sparse:
        :arg words:
        :return:
        """

        count = {}

        for word in words:
            if word in count:
                count[word] += 1
            else:
                count[word] = 1

        if sparse:
            vector = {}
        else:
            vector = np.zeros((len(self.vocab),), )
        if mode == 'tfidf':
            for w, c in count.items():
                tf = 1 + np.log(c)
                idf = np.log(1 + self.document_count /
                             (1 + self.df[w]))
                vector[w] = tf / idf

            return self.normalize_l2_sparse_vector(vector, sparse=sparse)
        elif mode == 'binary':
            x = np.zeros((len(count)))
            for w, c in count.items():
                x[w] = 1
            return x



    @staticmethod
    def normalize_l2_sparse_vector(vector, sparse):
        if sparse:
            sum_square = 0.0
            for key in vector:
                sum_square += vector[key] * vector[key]
            result = {}
            for key in vector:
                result[key] = vector[key] / np.sqrt(sum_square)
        else:
            result = vector / np.linalg.norm(vector)
            # result = sklearn.preprocessing.normalize(vector)
        return result


tfidf = TfIdf()


def text_to_vector(words: List[int]):
    # assert isinstance(words, list) and isinstance(words[0], int)
    assert isinstance(words, list)
    return tfidf.words_to_vector(words)


if __name__ == '__main__':
    for d, l in tfidf.train_data[:2]:
        print(d)
        print(tfidf.words_to_vector(d))
