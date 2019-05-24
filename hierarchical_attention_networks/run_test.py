from .train import eval_model, prepare_model
import torch
from .load_data import load_data
import os
from sklearn.metrics import classification_report
from .utils import *


BATCH_SIZE = 64


def key_func(fpath):
    acc = fpath.split('_')[1]
    acc = acc.replace('acc', '')
    return float(acc)


# def key_func(fpath):
#     acc = fpath.split('_')[2]
#     acc = acc.replace('loss', '').replace('.pth', '')
#     return -float(acc)


def get_best_model(folder_path):
    models = list(p for p in os.listdir(folder_path) if p.endswith('.pth'))
    sorted_models = sorted(models, key=key_func, reverse=True)
    # sorted_models = sorted(models, key=key_func, reverse=False)
    print(sorted_models[:2])

    return os.path.join(folder_path, sorted_models[0])


def run_test(save_dir):
    args = load_args(save_dir)
    print('args', vars(args))
    mpath = get_best_model(save_dir)
    print('best model', mpath)

    batch_size = BATCH_SIZE
    torch.device('cuda:0')
    TEXT, LABEL, vocab_size, word_embeddings, \
    train_iter, valid_iter, test_iter = load_data(train_bsize=batch_size,
                                                  bsize=batch_size * 2,
                                                  embedding_length=args.emb_size)
    print(LABEL.vocab.stoi)
    print(TEXT.vocab.vectors.size())
    state_dict = torch.load(mpath)
    print(state_dict['word_embeddings.weight'].size())
    assert state_dict['word_embeddings.weight'].size() == TEXT.vocab.vectors.size()

    print(state_dict.keys())

    model, _ = prepare_model(args, 2, word_embeddings)

    model.cuda()
    model.load_state_dict(state_dict)
    model.eval()
    _, _, y_gt, y_prediction = eval_model(model, test_iter, args)
    print(classification_report(y_gt, y_prediction, digits=4, labels=[0, 1],
                                target_names=['Adopted', 'Unadopted']))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--batchsize", type=int, default=64)

    args = parser.parse_args()

    BATCH_SIZE = args.batchsize
    run_test(args.save_dir)
