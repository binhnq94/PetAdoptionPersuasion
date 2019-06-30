import torch
import torch.nn as nn
from .model import LstmLayer, MultiAttentionLayer, make_tokens_att_full, \
    compute_loss_from_att_weights, compute_loss, filter_sents


class LayerOne(nn.Module):

    def __init__(self, embedding_size, embedding_weight, attention_hops, lstm_hidden_size=256,
                 lstm_num_layers=1, attention_size=64, custom_loss=False):
        super(LayerOne, self).__init__()
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

    def sentence_level(self, flat_emb_document, flat_sequence_lengths, document_lengths, origin_shape):
        tokens_lstm = self.lstm_layers[0](flat_emb_document, flat_sequence_lengths)
        tokens_att, att_weights_0 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)

        tokens_att = make_tokens_att_full(tokens_att, document_lengths, origin_shape[0], origin_shape[1])

        assert tokens_att.shape[-1] == self.lstm_hidden_size * 2

        tokens_att_layer_one, att_weights_01 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)
        tokens_att_layer_one = make_tokens_att_full(tokens_att_layer_one, document_lengths,
                                                    origin_shape[0], origin_shape[1])

        return tokens_lstm, tokens_att, tokens_att_layer_one

    def document_level(self, tokens_att, document_lengths):
        sents_lstm = self.lstm_layers[1](tokens_att, document_lengths)
        sents_att, att_weights_1 = self.att_layers[1](sents_lstm, document_lengths)
        return sents_att

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

        flat_document, flat_sequence_lengths = filter_sents(flat_document, flat_sequence_lengths)
        flat_emb_document = self.word_embeddings(flat_document)

        tokens_lstm, tokens_att, tokens_att_layer_one = self.sentence_level(flat_emb_document,
                                                                            flat_sequence_lengths,
                                                                            document_lengths,
                                                                            origin_shape)

        sents_att = self.document_level(tokens_att, document_lengths)

        # TODO: do custom_loss
        # if self.custom_loss:
        #     custom_loss = compute_loss(att_weights_0, att_weights_1, document_lengths)
        #     return sents_att, custom_loss
        return tokens_lstm, tokens_att_layer_one, sents_att


class LayerTwo(nn.Module):

    def __init__(self, input_size, attention_hops, lstm_hidden_size=256,
                 lstm_num_layers=1, attention_size=64, custom_loss=False):
        super(LayerTwo, self).__init__()
        self.custom_loss = custom_loss
        self.attention_hops = attention_hops
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = nn.ModuleList([
            LstmLayer(input_size, lstm_hidden_size, num_layers=lstm_num_layers),
            LstmLayer(2 * lstm_hidden_size, lstm_hidden_size, num_layers=lstm_num_layers)
        ])
        self.att_layers = nn.ModuleList([
            MultiAttentionLayer(lstm_hidden_size * 2, attention_size=attention_size,
                                attention_hops=self.attention_hops[0]),
            MultiAttentionLayer(lstm_hidden_size * 2, attention_size=attention_size,
                                attention_hops=self.attention_hops[1])
        ])

    def forward(self, tokens_lstm_layer_one, tokens_att_layer_one, sents_att_layer_one, document, document_lengths,
                sequence_lengths):
        """

        :param document: [batch_size, max_sent, max_word]
        :param document_lengths: [batch_size]
        :param sequence_lengths: [batch_size, max_seq]
        :return:
        """
        origin_shape = document.shape
        # flat_document = document.view(-1, origin_shape[-1])
        flat_sequence_lengths = sequence_lengths.view(-1)

        flat_emb_document = tokens_lstm_layer_one

        tokens_lstm = self.lstm_layers[0](flat_emb_document, flat_sequence_lengths)

        # TODO: element-wise product tokens_att_layer_one and tokens_lstm

        tokens_att, att_weights_0 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)

        tokens_att = make_tokens_att_full(tokens_att, document_lengths, origin_shape[0], origin_shape[1])

        assert tokens_att.shape[-1] == self.lstm_hidden_size * 2

        sents_lstm = self.lstm_layers[1](tokens_att, document_lengths)

        # TODO: element-wise product sents_lstm and sents_att_layer_one

        sents_att, att_weights_1 = self.att_layers[1](sents_lstm, document_lengths)

        # TODO: do custom_loss
        # if self.custom_loss:
        #     custom_loss = self.compute_loss(att_weights_0, att_weights_1, document_lengths)
        #     return sents_att, custom_loss
        return sents_att


class MultiReasoning:

    def __init__(self, args):

        self.layer_one = LayerOne(args.embedding_size, embedding_weight, args.attention_hops,
                                  lstm_hidden_size=args.lstm_hidden_size,
                                  lstm_num_layers=args.lstm_num_layers, attention_size=args.attention_size,
                                  custom_loss=args.custom_loss)

        self.drop_out = args.drop_out
        if self.drop_out > 0:
            self.drop_out_layer = nn.Dropout(self.drop_out)
        self.fc_layer = nn.Linear(2 * lstm_hidden_size, fc_size)
        self.final_linear = nn.Linear(fc_size, output_size)

    def compute_fc_layers(self, x):
        if self.drop_out > 0:
            x = self.drop_out_layer(x)

        x = self.fc_layer(x)
        x = self.final_linear(x)
        return x

    def forward(self, document, document_lengths, sequence_lengths):
        tokens_lstm, tokens_att_for_layer_two, sents_att = self.layer_one(document, document_lengths, sequence_lengths)
