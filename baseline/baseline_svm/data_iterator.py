import random
import numpy as np
from .unigram_bigram import text_to_vector, tfidf
from .create_w2v import load_w2v

LABELS = ['Unadopted', 'Adopted']


class GenData:

    def __init__(self, data_file, data_kind, shuffle=True):
        self.data_kind = data_kind
        w2v = load_w2v(data_kind)
        data = self.load_all_data(data_file)
        assert len(w2v) == len(data)
        data = self.convert_data_to_matrix(data)

        # self.data = np.hstack(data, w2v)
        self.data = w2v, data
        print('data.shape', self.data[0].shape, self.data[1].shape)

        self.number = len(self.data)
        self.shuffle = shuffle
        self.indexes = list(range(self.number))
        if self.shuffle:
            random.shuffle(self.indexes)
        self.cur_index_batch = 0

    def convert_data_to_matrix(self, data):

        # x_vectors = []
        y_vectors = []
        for idx, (text, label) in enumerate(data):
            x_vec = text_to_vector(text)
            # x_vectors.append(x_vec)
            y_vectors.append(LABELS.index(label))

        return np.array(y_vectors, dtype=np.int)

    @staticmethod
    def load_all_data(data_file):
        data = []
        with open(data_file, encoding='utf-8') as fi:

            for index_line, line in enumerate(fi):
                line = line.strip()
                if line:
                    _id, text, label = line.split('\t')
                    text = tfidf.preprocess_sen(text)

                    # TODO: fix it
                    # if text:
                    data.append((text, label))
        return data


if __name__ == '__main__':
    GenData('datasets/converted-v2_train_petfinder_data_study2.csv', data_kind='train')
