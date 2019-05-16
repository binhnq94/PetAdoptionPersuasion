from .train import eval_model
import torch
from .load_data import load_data
import os
from sklearn.metrics import classification_report


def key_func(fpath):
    acc = fpath.split('_')[1]
    acc = acc.replace('acc', '')
    return float(acc)


# def key_func(fpath):
#     acc = fpath.split('_')[2]
#     acc = acc.replace('loss', '').replace('.pth', '')
#     return -float(acc)


def get_best_model(folder_path):
    models = list(os.listdir(folder_path))
    sorted_models = sorted(models, key=key_func, reverse=True)
    # sorted_models = sorted(models, key=key_func, reverse=False)
    print(sorted_models[:2])

    return os.path.join(folder_path, sorted_models[0])


def run_test(save_dir, args):
    print('args', vars(args))
    mpath = get_best_model(save_dir)
    print('best model', mpath)

    batch_size = 16
    # embedding_length = 200
    # embedding_length = 300
    embedding_length = args.emb_size

    torch.device('cuda:0')

    TEXT, LABEL, vocab_size, word_embeddings, \
    train_iter, valid_iter, test_iter = load_data(train_bsize=batch_size,
                                                  bsize=batch_size * 2,
                                                  embedding_length=embedding_length)

    # torch.save(TEXT.vocab.stoi, 'models/TEXT.stoi.pt')
    # torch.save(LABEL.vocab.stoi, 'models/LABEL.stoi.pt')

    # torch.save(TEXT, 'models/TEXT.field.pt')
    # torch.save(LABEL, 'models/LABEL.field.pt')
    print(LABEL.vocab.stoi)

    print(TEXT.vocab.vectors.size())
    state_dict = torch.load(mpath)
    print(state_dict['word_embeddings.weight'].size())
    assert state_dict['word_embeddings.weight'].size() == TEXT.vocab.vectors.size()

    print(state_dict.keys())

    if args.model == 'hierarchical':
        from .model import HierarchicalAttention
        model = HierarchicalAttention(output_size=2,
                                      embedding_size=embedding_length,
                                      embedding_weight=word_embeddings,
                                      lstm_hidden_size=256)
    elif args.model == 'multi_att':
        from .model import HierarchicalMultiAttention

        model = HierarchicalMultiAttention(output_size=2,
                                           embedding_size=embedding_length,
                                           embedding_weight=word_embeddings,
                                           lstm_hidden_size=256,
                                           custom_loss=args.custom_loss,
                                           num_att=10)

    model.cuda()

    # # model = LSTMClassifier(*args, **kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    _, _, y_gt, y_prediction = eval_model(model, test_iter)
    print(classification_report(y_gt, y_prediction, digits=4, labels=[0, 1],
                                target_names=['Adopted', 'Unadopted']))

    # test_model(model, valid_iter)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--emb_size", type=int, default=200, help='Embedding size')
    parser.add_argument('--model', type=str, choices=['hierarchical', 'multi_att'], default='hierarchical')
    parser.add_argument('--custom_loss', action='store_true', help='Using custom loss')

    args = parser.parse_args()

    run_test(args.save_dir, args)
