import torch
from .load_data import load_data
import torch.nn.functional as F
import os
import datetime
import numpy as np
import random
from .utils import *

SEED = 0

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_DIR = os.path.dirname(__file__)
BASENAME_DIR = os.path.basename(CURRENT_DIR)


DEBUG = False


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def get_data_from_batch(batch):
    document, document_lengths, sent_lengths = batch.document
    target = batch.label
    target = torch.autograd.Variable(target).long()
    if torch.cuda.is_available():
        document = document.cuda()
        document_lengths = document_lengths.cuda()
        sent_lengths = sent_lengths.cuda()
        target = target.cuda()
    return document, document_lengths, sent_lengths, target


def train_model(model, train_iter, optim, epoch, penalty_ratio):
    total_epoch_loss = 0

    count_true = 0
    count_all = 0

    model.cuda()
    model.train()
    for idx, batch in enumerate(train_iter):
        document, document_lengths, sent_lengths, target = get_data_from_batch(batch)
        optim.zero_grad()
        prediction = model(document, document_lengths, sent_lengths)

        if model.custom_loss:
            prediction, custom_loss = prediction

        loss = loss_fn(prediction, target)
        cross_entropy_loss = loss

        if model.custom_loss:
            if DEBUG:
                print("cross_entropy_loss", loss.item(), "custom_loss", custom_loss.item())
            loss = loss + penalty_ratio*custom_loss
            if torch.isnan(loss).sum() > 0:
                print("cross_entropy_loss", loss.item(), "custom_loss", custom_loss.item())
                print("document.shape", document.shape)
                print("document_lengths", document_lengths)
                print("sent_lengths", sent_lengths)
                print("target", target)
                raise ValueError("Loss = NaN")

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        # acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()

        total_epoch_loss += loss.item()
        count_true += num_corrects
        count_all += len(batch)

        if (idx + 1) % (len(train_iter) // 5) == 0:
            print("DEBUG:", "cross_entropy_loss", cross_entropy_loss.item(), "custom_loss", custom_loss.item())
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, '
                  f'Training Accuracy: {count_true / count_all: .4f}%')

    return total_epoch_loss / len(train_iter), count_true / count_all


def loss_fn(prediction, target):
    return F.cross_entropy(prediction, target)


def eval_model(model, data_iter, penalty_ratio):
    total_epoch_loss = 0

    y_prediction = []
    y_gt = []

    count_true = 0
    count_all = 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_iter):
            document, document_lengths, sent_lengths, target = get_data_from_batch(batch)
            prediction = model(document, document_lengths, sent_lengths)
            if model.custom_loss:
                prediction, custom_loss = prediction
            loss = loss_fn(prediction, target)
            if model.custom_loss:
                loss = loss + penalty_ratio*custom_loss

            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            # acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()

            y_prediction.extend(torch.max(prediction, 1)[1].view(target.size()).tolist())
            y_gt.extend(target.tolist())

            count_true += num_corrects.item()
            count_all += len(batch)

    # return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
    return total_epoch_loss / len(data_iter), count_true / count_all, y_gt, y_prediction


def prepare_save_dir(args):
    save_dir = f"{CURRENT_DIR}/models/{args.model}_{datetime.datetime.now().strftime('%y%m%d%-H%M%S'):}"
    # prepare save_dir
    assert not os.path.exists(save_dir)
    print("SAVE_DIR", save_dir)
    os.makedirs(save_dir)

    save_args(args, save_dir)
    return save_dir


def prepare_model(args, output_size, word_embeddings):
    if args.model == 'hierarchical':
        from .model import HierarchicalAttention
        model = HierarchicalAttention(output_size=output_size,
                                      embedding_size=args.emb_size,
                                      embedding_weight=word_embeddings,
                                      lstm_hidden_size=args.lstm_h_size)
    elif args.model == 'multi_att':
        from .model import HierarchicalMultiAttention
        model = HierarchicalMultiAttention(output_size=output_size,
                                           embedding_size=args.emb_size,
                                           embedding_weight=word_embeddings,
                                           lstm_hidden_size=args.lstm_h_size,
                                           custom_loss=args.custom_loss,
                                           attention_hops=10)
    else:
        raise ValueError('Model kind = {}'.format(args.model))

    if args.optim == 'adam':
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()))
    return model, optim


def main(args):
    print("args", vars(args))
    save_dir = prepare_save_dir(args)
    output_size = 2

    TEXT, LABEL, vocab_size, word_embeddings, \
    train_iter, valid_iter, test_iter = load_data(train_bsize=args.batchsize,
                                                  bsize=args.batchsize * 2,
                                                  embedding_length=args.emb_size)
    print('LABEL.vocab.stoi', LABEL.vocab.stoi)

    # LOAD MODEL
    torch.device('cuda:0')

    model, optim = prepare_model(args, output_size, word_embeddings)

    if torch.cuda.is_available():
        model.cuda()

    if args.pretrained is not None:
        raise NotADirectoryError
        print('Load model from:', args.pretrained)
        state_dict = torch.load(args.pretrained)
        model.load_state_dict(state_dict)

    print("Start Train")
    for epoch in range(args.epoch):
        train_loss, train_acc = train_model(model, train_iter, optim, epoch, args.penalty_ratio)
        val_loss, val_acc, _, _ = eval_model(model, valid_iter, args.penalty_ratio)

        print("Epoch:{:4d}, loss:{}, acc:{}, "
              "val_loss:{}, val_acc {}".format(epoch + 1, train_loss, train_acc, val_loss, val_acc))

        save_path = f'{save_dir}/epoch:{epoch + 1:04d}_acc{val_acc:4f}_loss{val_loss:4f}.pth'

        print(f'Saving model to {save_path}')
        torch.save(model.state_dict(), save_path)

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=25)

    parser.add_argument('--model', type=str, choices=['hierarchical', 'multi_att'], default='hierarchical')
    parser.add_argument("--emb_size", type=int, default=200, help='Embedding size')
    parser.add_argument('--lstm_h_size', type=int, default=256, help='LSTM size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of lstm layers')
    parser.add_argument("--att_size", type=int, default=64, help='Attention size')
    parser.add_argument('--att_hops', type=int, default=10, help='Number attention hops')
    parser.add_argument('--custom_loss', action='store_true', help='Using custom loss')
    parser.add_argument('--penalty_ratio', type=float, default=0.03, help='Lambda of custom_loss')

    parser.add_argument('--optim', type=str, choices=['adam', 'rmsprop'], default='rmsprop')

    parser.add_argument('--pretrained', type=str, default=None)

    args_ = parser.parse_args()
    save_dir_ = main(args_)
    from .run_test import run_test
    run_test(save_dir_)
