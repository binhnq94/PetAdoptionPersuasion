import torch
import torch.nn as nn
from .model import LstmLayer, MultiAttentionLayer, make_tokens_att_full, \
    compute_loss_from_att_weights, compute_loss, filter_sents


class LayerOne(nn.Module):

    def __init__(self, embedding_size, embedding_weight, attention_hops, lstm_hidden_size=256,
                 lstm_num_layers=1, attention_size=64,
                 # custom_loss=False
                 ):
        super(LayerOne, self).__init__()
        # self.custom_loss = custom_loss
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

        sentences_present, att_weights_0 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)
        sentences_present = make_tokens_att_full(sentences_present, document_lengths, origin_shape[0], origin_shape[1])

        assert sentences_present.shape[-1] == self.lstm_hidden_size * 2

        sentences_present_layer_one, att_weights_01 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)
        # sentences_present_layer_one = make_tokens_att_full(sentences_present_layer_one, document_lengths,
        #                                                    origin_shape[0], origin_shape[1])

        return tokens_lstm, sentences_present, sentences_present_layer_one

    def document_level(self, sentences_present, document_lengths):
        sents_lstm = self.lstm_layers[1](sentences_present, document_lengths)
        documents_present, att_weights_1 = self.att_layers[1](sents_lstm, document_lengths)
        return documents_present

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

        tokens_lstm, sentences_present, sentences_present_layer_one = self.sentence_level(flat_emb_document,
                                                                                          flat_sequence_lengths,
                                                                                          document_lengths,
                                                                                          origin_shape)

        documents_present = self.document_level(sentences_present, document_lengths)

        # TODO: do custom_loss
        # if self.custom_loss:
        #     custom_loss = compute_loss(att_weights_0, att_weights_1, document_lengths)
        #     return sents_att, custom_loss
        return tokens_lstm, sentences_present_layer_one, documents_present


class LayerTwo(nn.Module):

    def __init__(self, input_size, attention_hops, lstm_hidden_size=256,
                 lstm_num_layers=1, attention_size=64,
                 # custom_loss=False
                 ):
        super(LayerTwo, self).__init__()
        # self.custom_loss = custom_loss
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

    def forward(self, tokens_lstm_layer_one, sentences_present_layer_one, documents_present_layer_one,
                document, document_lengths, sequence_lengths):
        """

        :param documents_present_layer_one: [batch_size]
        :param sentences_present_layer_one: [batch_size*max_sentence, vector_size]
        :param tokens_lstm_layer_one:
        :param document: [batch_size, max_sent, max_word]
        :param document_lengths: [batch_size]
        :param sequence_lengths: [batch_size, max_seq]
        :return:
        """
        origin_shape = document.shape
        flat_sequence_lengths = sequence_lengths.view(-1)

        # print(tokens_lstm_layer_one.shape)
        # print(flat_sequence_lengths.tolist())

        flat_sequence_lengths = flat_sequence_lengths[flat_sequence_lengths > 0]

        tokens_lstm = self.lstm_layers[0](tokens_lstm_layer_one, flat_sequence_lengths)

        # TODO: element-wise product tokens_att_layer_one and tokens_lstm
        # print("layertwo", tokens_lstm.shape, sentences_present_layer_one.shape)
        tokens_lstm = tokens_lstm * sentences_present_layer_one.unsqueeze(1)

        sentences_present, att_weights_0 = self.att_layers[0](tokens_lstm, flat_sequence_lengths)
        sentences_present = make_tokens_att_full(sentences_present, document_lengths, origin_shape[0], origin_shape[1])

        assert sentences_present.shape[-1] == self.lstm_hidden_size * 2

        sents_lstm = self.lstm_layers[1](sentences_present, document_lengths)

        # TODO: element-wise product sents_lstm and sents_att_layer_one
        # print("layertwo shape", sents_lstm.shape, documents_present_layer_one.shape)
        sents_lstm = sents_lstm * documents_present_layer_one.unsqueeze(1)

        documents_present, att_weights_1 = self.att_layers[1](sents_lstm, document_lengths)

        # TODO: do custom_loss
        # if self.custom_loss:
        #     custom_loss = self.compute_loss(att_weights_0, att_weights_1, document_lengths)
        #     return sents_att, custom_loss
        return documents_present


class MultiReasoning(nn.Module):

    def __init__(self, args, embedding_weight, output_size):
        super(MultiReasoning, self).__init__()

        self.custom_loss = False
        self.layer_one = LayerOne(args.emb_size, embedding_weight,
                                  attention_hops=args.att_hops,
                                  lstm_hidden_size=args.lstm_h_size,
                                  lstm_num_layers=args.lstm_layers,
                                  attention_size=args.att_size,
                                  # custom_loss=args.custom_loss
                                  )
        self.layer_two = LayerTwo(2 * args.lstm_h_size,
                                  attention_hops=args.att_hops,
                                  lstm_hidden_size=args.lstm_h_size,
                                  lstm_num_layers=args.lstm_layers,
                                  attention_size=args.att_size,
                                  # custom_loss=args.custom_loss
                                  )

        self.drop_out = args.drop_out
        if self.drop_out > 0:
            self.drop_out_layer = nn.Dropout(self.drop_out)
        self.fc_layer = nn.Linear(2 * args.lstm_h_size, args.fc_size)
        self.final_linear = nn.Linear(args.fc_size, output_size)

    def compute_fc_layers(self, x):
        if self.drop_out > 0:
            x = self.drop_out_layer(x)

        x = self.fc_layer(x)
        x = self.final_linear(x)
        return x

    def forward(self, document, document_lengths, sequence_lengths):
        tokens_lstm_layer_one, sentences_present_layer_one, documents_present_layer_one = self.layer_one(
            document,
            document_lengths,
            sequence_lengths)

        documents_present = self.layer_two(tokens_lstm_layer_one, sentences_present_layer_one,
                                           documents_present_layer_one, document, document_lengths, sequence_lengths)

        final_outputs = self.compute_fc_layers(documents_present)

        return final_outputs
