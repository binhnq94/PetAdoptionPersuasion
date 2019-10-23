import torch
from baseline_bilstm.load_data import load_data
import torch.nn.functional as F
from sklearn.metrics import classification_report
import os
import datetime
import numpy as np
import random
from .model import CNN_Text
from hierarchical_attention_networks.utils import *
import time

SEED = 0

CURRENT_DIR = "/media/binhnq/New Volume1"
BASENAME_DIR = os.path.basename(CURRENT_DIR)

DEBUG = False


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch, optim):
    total_epoch_loss = 0

    count_true = 0
    count_all = 0

    model.cuda()
    model.train()
    optim.zero_grad()

    for idx, batch in enumerate(train_iter):
        text, lengths = batch.text
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            lengths = lengths.cuda()
            target = target.cuda()
        # if (text.size()[0] is not 32):  # One of the batch returned by BucketIterator has length different than 32.
        #     continue
        optim.zero_grad()
        # prediction = model(text, lengths, len(batch))
        prediction = model(text)

        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        # acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()

        total_epoch_loss += loss.item()
        count_true += num_corrects
        count_all += len(batch)

        if (idx+1) % (len(train_iter) // 5) == 0:
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, '
                  f'Training Accuracy: {count_true/count_all: .4f}%')

    return total_epoch_loss / len(train_iter), count_true / count_all


def loss_fn(prediction, target):
    return F.cross_entropy(prediction, target)


def eval_model(model, test_iter):
    total_epoch_loss = 0

    y_prediction = []
    y_gt = []

    count_true = 0
    count_all = 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            text, lengths = batch.text
            # if (text.size()[0] is not 32):
            #     continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                lengths = lengths.cuda()
                target = target.cuda()
            # prediction = model(text, lengths, len(batch))
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            # acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()

            y_prediction.extend(torch.max(prediction, 1)[1].view(target.size()).tolist())
            y_gt.extend(target.tolist())

            count_true += num_corrects.item()
            count_all += len(batch)

    print(classification_report(y_gt, y_prediction, digits=4, labels=[0, 1],
                                target_names=['Adopted', 'Unadopted']))

    # return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
    return total_epoch_loss/len(test_iter), count_true/count_all



def prepare_save_dir(args):
    save_dir = f"{CURRENT_DIR}/models/{args.model}_{datetime.datetime.now().strftime('%y%m%d%H%M%S'):}"
    # prepare save_dir
    assert not os.path.exists(save_dir)
    print("SAVE_DIR", save_dir)
    os.makedirs(save_dir)

    save_args(args, save_dir)
    return save_dir

def main(args):
    print("args", vars(args))
    print('batch_size', args.batch_size)
    save_dir = prepare_save_dir(args)
    output_size = 2

    TEXT, LABEL, vocab_size, word_embeddings, \
    train_iter, valid_iter, test_iter = load_data(train_bsize=args.batch_size,
                                                  bsize=args.batch_size * 2,
                                                  embedding_length=args.emb_size)
    print('LABEL.vocab.stoi', LABEL.vocab.stoi)

    # LOAD MODEL
    torch.device('cuda:0')

    model = CNN_Text(args, word_embeddings, output_size)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    if torch.cuda.is_available():
        model.cuda()

    if args.pretrained is not None:
        raise NotADirectoryError
        print('Load model from:', args.pretrained)
        state_dict = torch.load(args.pretrained)
        model.load_state_dict(state_dict)

    for epoch in range(args.epoch):
        train_loss, train_acc = train_model(model, train_iter, epoch, optim)
        val_loss, val_acc = eval_model(model, valid_iter)

        print("Epoch:{:4d}, loss:{}, acc:{}, "
              "val_loss:{}, val_acc {}".format(epoch+1, train_loss, train_acc, val_loss, val_acc))

        save_path = f'{save_dir}/epoch:{epoch+1:04d}_acc{val_acc:4f}_loss{val_loss:4f}.pth'

        print(f'Saving model to {save_path}')
        torch.save(model.state_dict(), save_path)

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=25)

    parser.add_argument('--model', type=str, choices=['cnn'], default='cnn')
    parser.add_argument("--emb_size", type=int, default=200, help='Embedding size')
    parser.add_argument('--kernel_sizes', type=list, default=[2, 3, 4, 5])
    parser.add_argument('--kernel_num', type=int, default=100)
    parser.add_argument('--fc_size', type=int, default=128, help='Full connected size.')

    parser.add_argument('--drop_out', default=0.0, type=float, help='Drop out for last fc')
    parser.add_argument('--optim', type=str, choices=['adam', 'rmsprop'], default='rmsprop')
    parser.add_argument('--pretrained', type=str, default=None)

    args_ = parser.parse_args()

    SEED = args_.seed

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = main(args_)
    from .run_test import run_test
    run_test(save_dir)
