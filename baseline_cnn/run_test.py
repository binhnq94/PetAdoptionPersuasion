from .train import eval_model
from .model import CNN_Text
import torch
from baseline_bilstm.load_data import load_data
import os
from hierarchical_attention_networks.utils import *


def key_func(fpath):
    acc = fpath.split('_')[1]
    acc = acc.replace('acc', '')
    return float(acc)


def get_best_model(folder_path):
    models = list(p for p in os.listdir(folder_path) if p.endswith('.pth'))

    sorted_models = sorted(models, key=key_func, reverse=True)
    print(sorted_models[:2])

    return os.path.join(folder_path, sorted_models[0])


def run_test(save_dir):
    args = load_args(save_dir)
    print('args', vars(args))

    mpath = get_best_model(save_dir)
    print('best model', mpath)

    batch_size = 64
    output_size = 2
    torch.device('cuda:0')

    TEXT, LABEL, vocab_size, word_embeddings, \
    train_iter, valid_iter, test_iter = load_data(train_bsize=batch_size,
                                                  bsize=batch_size * 2,
                                                  embedding_length=args.emb_size)

    print(LABEL.vocab.stoi)

    print(TEXT.vocab.vectors.size())
    state_dict = torch.load(mpath)
    # print(state_dict['word_embeddings.weight'].size())
    # assert state_dict['word_embeddings.weight'].size() == TEXT.vocab.vectors.size()

    print(state_dict.keys())
    model = CNN_Text(args, word_embeddings, output_size)
    model.cuda()

    # # model = LSTMClassifier(*args, **kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    print('test')
    eval_model(model, test_iter)
    # test_model(model, valid_iter)
    print('val')
    eval_model(model, valid_iter)



if __name__ == '__main__':
    import sys

    run_test(sys.argv[1])
