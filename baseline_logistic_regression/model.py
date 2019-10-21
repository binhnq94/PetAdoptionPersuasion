import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LogisticRegression(torch.nn.Module):
    def __init__(self, args, embedding_weight, output_size):
        super(LogisticRegression, self).__init__()
        self.args = args
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weight)
        self.vocab_size = embedding_weight.shape[0]

        if args.binary:
            self.linear = torch.nn.Linear(args.emb_size + self.vocab_size, 1)
        else:
            self.linear = torch.nn.Linear(args.emb_size, 1)


    def forward(self, x):
        batch_size = x.shape[0]
        if self.args.binary:
            x1 = torch.zeros(batch_size, self.vocab_size).cuda()
            for i in range(batch_size):
                x1[i, x[i]] = 1

        x = self.word_embeddings(x)     # [batch_size, num_word, emb_size]

        x = torch.mean(x, dim=(1,))
        if self.args.binary:
            x = torch.cat([x, x1], dim=1)
        y_pred = torch.sigmoid(self.linear(x))
        # print(y_pred.shape)
        return y_pred
