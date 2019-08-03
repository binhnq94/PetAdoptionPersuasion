import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args, embedding_weight, output_size):
        super(CNN_Text, self).__init__()

        Ci = 1

        # self.embed = nn.Embedding.from_pretrained(embedding_weight)
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weight)

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, args.kernel_num, (kernel_size, args.emb_size))
                                     for kernel_size in args.kernel_sizes])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.drop_out = args.drop_out
        if self.drop_out > 0:
            self.drop_out_layer = nn.Dropout(self.drop_out)
        self.fc_layer = nn.Linear(len(args.kernel_sizes)*args.kernel_num, args.fc_size)
        self.final_linear = nn.Linear(args.fc_size, output_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def compute_fc_layers(self, x):
        if self.drop_out > 0:
            x = self.drop_out_layer(x)

        x = self.fc_layer(x)
        x = self.final_linear(x)
        return x

    def forward(self, x):
        x = self.word_embeddings(x)  # (N, W, D)
        
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        logit = self.compute_fc_layers(x)  # (N, C)
        return logit
