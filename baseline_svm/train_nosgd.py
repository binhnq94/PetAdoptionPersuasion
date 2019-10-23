import os
import pickle
from .data_iterator import GenData
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score
import datetime
from .file_utils import CURRENT_DIR


save_dir = f"baseline_tfidf_svm/models/no-sgd_svc_{datetime.datetime.now().strftime('%y%m%d%-H%M%S'):}"
assert not os.path.exists(save_dir)
os.makedirs(save_dir)


def save_model(cls, name="svm.pickle"):

    model_fpath = os.path.join(save_dir, name)
    print(f"Saving model to {model_fpath}")

    with open(model_fpath, 'wb') as fo:
        pickle.dump(cls, fo)


TRAIN_DATA_PATH = os.path.join(CURRENT_DIR, 'resources/train_data.pickle')
VAL_DATA_PATH = os.path.join(CURRENT_DIR, 'resources/val_data.pickle')
TEST_DATA_PATH = os.path.join(CURRENT_DIR, 'resources/test_data.pickle')


def build_data_pickle(data_path, data_kind, out_path):
    print(data_path, out_path)
    gen = GenData(data_path, data_kind)
    x, y = gen.data

    print(x.shape, y.shape)

    with open(out_path, 'wb') as fo:
        pickle.dump(gen.data, fo)

    del gen


def train():
    if not os.path.isfile(TRAIN_DATA_PATH):
        build_data_pickle("datasets/190524/converted-v2_train.csv", 'train', TRAIN_DATA_PATH)
        build_data_pickle("datasets/190524/converted-v2_val.csv", 'val', VAL_DATA_PATH)
        build_data_pickle("datasets/190524/converted-v2_test.csv", 'test', TEST_DATA_PATH)

    with open(TRAIN_DATA_PATH, 'rb') as fi:
        train_data = pickle.load(fi)
    print("done load train")

    with open(VAL_DATA_PATH, 'rb') as fi:
        val_data = pickle.load(fi)
    print("done load val")

    with open(TEST_DATA_PATH, 'rb') as fi:
        test_data = pickle.load(fi)
    print("done load test")

    try_to_find_best_C(train_data, val_data, test_data)


def try_to_find_best_C(train_data, val_data, test_data):
    max_acc = 0
    c_max = 0
    gamma_max = 0
    for c in [1.0, 0.1, 0.01, 0.001, 0.0001]:
            gamma = None
        # for gamma in [1e-3, 1e-2, 1e-1, 1]:
            cls = LinearSVC(C=c, loss='hinge', verbose=1, max_iter=1000)
            # c = 0.1
            # gamma = 1
            # cls = SVC(C=c, max_iter=1000, gamma=gamma)
            # cls = SVC(max_iter=100)
            cls.fit(train_data[0], train_data[1])

            print('val')
            y_predict = cls.predict(val_data[0])
            print(classification_report(val_data[1], y_predict, target_names=["Unadopted", "Adopted"], digits=4))
            val_acc = accuracy_score(val_data[1], y_predict)
            print('val', 'C=', c, 'gamma', gamma, 'acc=', val_acc)

            print('test')
            y_predict = cls.predict(test_data[0])
            print(classification_report(test_data[1], y_predict, target_names=["Unadopted", "Adopted"], digits=4))
            acc = accuracy_score(test_data[1], y_predict)
            print('test', 'C=', c, 'gamma', gamma, 'acc=', acc)

            save_model(cls, name=f'svm_C={c}_gamma={gamma}_acc={val_acc}.pickle')

            if acc > max_acc:
                max_acc = acc
                c_max = c
                gamma_max = gamma

    print(max_acc, c_max, gamma_max)


if __name__ == '__main__':
    train()
