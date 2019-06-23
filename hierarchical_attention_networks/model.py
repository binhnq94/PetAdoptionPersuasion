import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import math


def create_mask(lengths, max_length):
    mask = torch.zeros(len(lengths), max_length)
    for k, length in enumerate(lengths):
        mask[k][:length] = torch.ones(length)
    return mask


class AttentionLayer(nn.Module):
    def __init__(self, input_size, attention_size=64):
        super(AttentionLayer, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.attention_size = attention_size
        self.in_dense_layer = torch.nn.Linear(self.input_size, self.attention_size)
        self.out_dense_layer = torch.nn.Linear(self.attention_size, 1, bias=False)

    def forward(self, seq_input, lengths):
        """

        :param seq_input: [batch_size, seq_len, emb]
        :param lengths: [batch_size, 1]
        :return:
        """

        out_dense = self.in_dense_layer(seq_input)
        out_dense = torch.tanh(out_dense)
        out_dense = self.out_dense_layer(out_dense)
        exps = torch.exp(out_dense)
        mask = create_mask(lengths, seq_input.shape[1]).to(self.device).unsqueeze(2)
        masked_exps = exps * mask
        sumed_exps = torch.sum(masked_exps, 1, keepdim=True)
        weights_att = masked_exps / sumed_exps

        out = torch.sum(weights_att * seq_input, 1)
        return out, weights_att.squeeze(2)


class LstmLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LstmLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        return h_0, c_0

    def forward(self, sequence_input, seq_lengths):
        """

        :param sequence_input:
        :param seq_lengths:
        :return: [batch_size, max_word, 2*rnn_size]
        """
        init_length = sequence_input.shape[1]
        batch_size = sequence_input.shape[0]

        sorted_seq_lengths, argsort_lengths = seq_lengths.sort(0, descending=True)

        # invert_argsort_lengths = torch.LongTensor(argsort_lengths.shape).fill_(0).to(self.device)
        invert_argsort_lengths = torch.LongTensor(argsort_lengths.shape).fill_(0).cuda()
        for i, v in enumerate(argsort_lengths):
            invert_argsort_lengths[v.data] = i

        sorted_input = sequence_input[argsort_lengths]
        sorted_input = pack_padded_sequence(sorted_input, sorted_seq_lengths, batch_first=True)

        lstm_outputs, _ = self.lstm(sorted_input, self.init_hidden(batch_size))

        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        lstm_outputs = lstm_outputs[invert_argsort_lengths]

        assert not lstm_outputs.shape[1] < init_length

        return lstm_outputs


class HierarchicalAttention(nn.Module):

    def __init__(self, output_size, embedding_size, embedding_weight, lstm_hidden_size=256,
                 lstm_num_layers=1, attention_size=64):
        super(HierarchicalAttention, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weight)

        self.lstm_layers_0 = LstmLayer(embedding_size, lstm_hidden_size, num_layers=lstm_num_layers)
        self.lstm_layers_1 = LstmLayer(lstm_hidden_size * 2, lstm_hidden_size, num_layers=lstm_num_layers)

        self.att_layers_0 = AttentionLayer(lstm_hidden_size * 2, attention_size)
        self.att_layers_1 = AttentionLayer(lstm_hidden_size * 2, attention_size)

        self.final_linear = nn.Linear(lstm_hidden_size * 2, output_size)

    def filter_sents(self, flat_document, flat_sequence_lengths):
        return flat_document[flat_sequence_lengths > 0], flat_sequence_lengths[flat_sequence_lengths > 0]

    def make_tokens_att_full(self, tokens_att, document_lengths, batch_size, max_sent):
        """Padding to all document have same len."""
        # print("tokens_att.shape", tokens_att.shape)
        out_tokens_att = torch.zeros(batch_size, max_sent, tokens_att.shape[-1]).cuda()
        # print("out_tokens.shape", out_tokens_att.shape)

        # print(document_lengths.tolist())

        pre_num_sent = 0
        for idx, num_sent in enumerate(document_lengths):
            # print("-----------")
            # print("idx, num_sent, max_sent, pre_num_sent", idx, num_sent.tolist(), max_sent, pre_num_sent)
            # print(idx*max_sent, idx*max_sent+num_sent)
            out_tokens_att[idx, :num_sent] = tokens_att[pre_num_sent:pre_num_sent + num_sent]
            pre_num_sent += num_sent
        return out_tokens_att

    def forward(self, document, document_lengths, sequence_lengths):
        """

        :param document: [batch_size, max_sent, max_word]
        :param document_lengths: [batch_size]
        :param sequence_lengths: [batch_size, max_seq]
        :return:
        """
        origin_shape = document.shape
        # print("document_shape", document.shape)
        flat_document = document.view(-1, origin_shape[-1])
        # print("flat_document.shape", flat_document.shape)

        # print("sequence_lengths", sequence_lengths.shape)
        flat_sequence_lengths = sequence_lengths.view(-1)
        # print("flat_sequence_lengths", flat_sequence_lengths.shape)

        flat_document, flat_sequence_lengths = self.filter_sents(flat_document, flat_sequence_lengths)

        flat_emb_document = self.word_embeddings(flat_document)
        # flat_emb_document = emb_document.view(-1, emb_document.shape[-2], emb_document.shape[-1])
        # print("flat_emb_document.shape", flat_emb_document.shape)

        tokens_lstm = self.lstm_layers_0(flat_emb_document, flat_sequence_lengths)
        tokens_att, _ = self.att_layers_0(tokens_lstm, flat_sequence_lengths)
        # print("tokens_att.shape", tokens_att.shape)
        tokens_att = self.make_tokens_att_full(tokens_att, document_lengths, origin_shape[0], origin_shape[1])
        # print("full_tokens_att.shape", tokens_att.shape)

        # tokens_att = tokens_att.view(emb_document.shape[0], emb_document.shape[1], -1)
        # assert tokens_att.shape[-1] == 256*2
        assert tokens_att.shape[-1] == self.lstm_hidden_size * 2

        sents_lstm = self.lstm_layers_1(tokens_att, document_lengths)
        sents_att, _ = self.att_layers_1(sents_lstm, document_lengths)
        # print("sents_att.shape", sents_att.shape)

        final_outputs = self.final_linear(sents_att)
        return final_outputs


class MultiAttentionLayer(nn.Module):

    def __init__(self, input_size, attention_size, attention_hops):
        super(MultiAttentionLayer, self).__init__()

        self.attention_hops = attention_hops
        self.linear_first = nn.Linear(input_size, attention_size, bias=False)
        self.linear_second = nn.Linear(attention_size, attention_hops, bias=False)

    def forward(self, seq_input, lengths):
        """

        :param seq_input: [bsize, max_len, ]
        :param lengths:
        :return:
        """
        batch_size, max_len = seq_input.shape[:-1]
        x = torch.tanh(self.linear_first(seq_input))  # [bsize, max_len, att_size]
        x = self.linear_second(x)  # [bsize, max_len, att_hops]
        attention = x.transpose(1, 2)  # [bsize, att_hops, max_len]

        mask = torch.arange(max_len).cuda().expand(batch_size, max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).repeat(1, self.attention_hops, 1)
        attention[~mask] = -float('inf')

        attention = attention.softmax(-1)  # [bsize, att_hops, max_len]
        seq_embedding = attention @ seq_input  # [bsize, att_hops, emb_size]

        avg_seq_embedding = seq_embedding.mean(1)

        return avg_seq_embedding, attention


class HierarchicalMultiAttention(HierarchicalAttention):

    def __init__(self, output_size, embedding_size, embedding_weight, attention_hops, lstm_hidden_size=256,
                 lstm_num_layers=1, attention_size=64, fc_size=128, drop_out=0, custom_loss=False,
                 use_transformer=False):
        super(HierarchicalAttention, self).__init__()
        self.custom_loss = custom_loss
        self.attention_hops = attention_hops
        self.lstm_hidden_size = lstm_hidden_size
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weight)
        self.lstm_layers = nn.ModuleList([
            LstmLayer(embedding_size, lstm_hidden_size, num_layers=lstm_num_layers),
            LstmLayer(2 * lstm_hidden_size, lstm_hidden_size, num_layers=lstm_num_layers)
        ])
        self.att_layers = nn.ModuleList([
            MultiAttentionLayer(lstm_hidden_size * 2, attention_size=attention_size,
                                attention_hops=self.attention_hops[0]),
            MultiAttentionLayer(lstm_hidden_size * 2, attention_size=attention_size,
                                attention_hops=self.attention_hops[1])
        ])
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformers = nn.ModuleList([
                TransformerLayer(lstm_hidden_size * 2),
                TransformerLayer(lstm_hidden_size * 2)
            ])
        self.drop_out = drop_out
        if self.drop_out > 0:
            self.drop_out_layer = nn.Dropout(self.drop_out)
        self.fc_layer = nn.Linear(2 * lstm_hidden_size, fc_size)
        self.final_linear = nn.Linear(fc_size, output_size)

    def compute_loss_from_att_weights(self, att_weights):
        tranpose_w = torch.transpose(att_weights, 1, 2)
        loss = (att_weights @ tranpose_w - torch.eye(att_weights.shape[1]).cuda()).norm(dim=(1, 2))
        return loss

    def compute_loss(self, att_weights0, att_weights1, document_lengths):
        loss_word_levels = self.compute_loss_from_att_weights(att_weights0)

        list_word_levels = []

        pre_index = 0
        for length in document_lengths:
            list_word_levels.append(loss_word_levels[pre_index:pre_index + length].mean().unsqueeze(0))
            pre_index += length

        sum_loss_word_levels = torch.cat(list_word_levels)

        # loss_sen_levels = self.compute_loss_from_att_weights(att_weights1) + sum_loss_word_levels
        # return loss_sen_levels.mean()
        return sum_loss_word_levels.mean(), self.compute_loss_from_att_weights(att_weights1).mean()

    def compute_fc_layers(self, x):
        if self.drop_out > 0:
            x = self.drop_out_layer(x)

        x = self.fc_layer(x)
        x = self.final_linear(x)
        return x

    def forward(self, document, document_lengths, sequence_lengths):
        """

        :param document: [batch_size, max_sent, max_word]
        :param document_lengths: [batch_size]
        :param sequence_lengths: [batch_size, max_seq]
        :return:
        """
        origin_shape = document.shape
        flat_document = document.view(-1, origin_shape[-1])
        flat_sequence_lengths = sequence_lengths.view(-1)

        flat_document, flat_sequence_lengths = self.filter_sents(flat_document, flat_sequence_lengths)
        flat_emb_document = self.word_embeddings(flat_document)

        tokens_lstm = self.lstm_layers[0](flat_emb_document, flat_sequence_lengths)
        if self.use_transformer:
            self.transformers[0](tokens_lstm, flat_sequence_lengths)

        tokens_att, att_weights_0 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)

        tokens_att = self.make_tokens_att_full(tokens_att, document_lengths, origin_shape[0], origin_shape[1])

        assert tokens_att.shape[-1] == self.lstm_hidden_size * 2

        sents_lstm = self.lstm_layers[1](tokens_att, document_lengths)
        if self.use_transformer:
            self.transformers[1](sents_lstm, document_lengths)
        sents_att, att_weights_1 = self.att_layers[1](sents_lstm, document_lengths)

        final_outputs = self.compute_fc_layers(sents_att)

        if self.custom_loss:
            custom_loss = self.compute_loss(att_weights_0, att_weights_1, document_lengths)
            return final_outputs, custom_loss
        return final_outputs


class TransformerLayer(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.input_size = input_size

        self.K = nn.Linear(input_size, input_size)
        self.Q = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, input_size)

        self.dropout = bool(dropout)
        if dropout:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, sequence_input, lengths):
        """

        :param sequence_input: [batch_size, max_word, emb_size]
        :param lengths: [batch_size, 1]
        :return:
        """
        batch_size, max_len = sequence_input.shape[:-1]
        # print('transformer', 'sequence_input.shape', sequence_input.shape)

        K = self.K(sequence_input)
        Q = self.Q(sequence_input)
        V = self.V(sequence_input)

        scores = Q@(K.transpose(-2, -1)) / math.sqrt(self.input_size)

        # print('transformer', 'scores.shape', scores.shape)

        mask = torch.arange(max_len).cuda().expand(batch_size, max_len) < lengths.unsqueeze(1)   # [batch_size, max_len]
        mask = mask.unsqueeze(1).repeat(1, max_len, 1)      # mask.unsqueeze(1) -> [batch_size, 1, max_len] -> [batch_size, max_len, max_len]
        # print('transformer', 'mask.shape', mask.shape)

        scores[~mask] = -float('inf')

        p_attn = scores.softmax(-1)

        if self.dropout:
            p_attn = self.dropout_layer(p_attn)

        x = p_attn @ V

        # print('transformer', 'x.shape', x.shape)

        return x
