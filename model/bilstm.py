import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .charbilstm import Charbilstm
from .charcnn import CharCNN

class Bilstm(nn.Module):

    def __init__(self, conf):
        super(Bilstm, self).__init__()
        self.device = conf.device
        self.vacab_size = conf.dicts['vocab_size']
        self.tag_size = conf.dicts['tag_size']
        self.embedding_dim = conf.embedding_dim
        input_size = self.embedding_dim
        self.char_feature_extractor = conf.char_feature_extractor
        self.hidden_dim = conf.hidden_dim
        if self.char_feature_extractor != 'none':
            input_size += conf.char_hidden_dim
            if self.char_feature_extractor == 'cnn':
                self.charcnn = CharCNN(conf)
            elif self.char_feature_extractor == 'bilstm':
                self.charbilstm = Charbilstm(conf)
        self.embed_layer = self.build_embed_layer(conf.embedding_table)
        self.word_drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(
            input_size,
            self.hidden_dim // 2,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.hidden_dim, self.tag_size)

    def build_embed_layer(self, embedding_table):
        if embedding_table is not None:
            embedding_table = torch.tensor(embedding_table, dtype=torch.float).to(self.device)
            embed_layer = nn.Embedding.from_pretrained(
                embedding_table,
                freeze=False
            )
        else:
            embed_layer = nn.Embedding(self.vacab_size, self.char_embedding_dim)
        return embed_layer

    def forward(self, data):
        word_seq_tensor = data['word_seq_tensor']
        word_seq_lens = data['word_seq_lens']

        embedding = self.word_drop(self.embed_layer(word_seq_tensor))
        if self.char_feature_extractor != 'none':
            if self.char_feature_extractor == 'cnn':
                char_feats = self.charcnn(data)
            elif self.char_feature_extractor == 'bilstm':
                char_feats = self.charbilstm(data)
            embedding = torch.cat([embedding, char_feats], dim=2)

        packed = pack_padded_sequence(
            embedding,
            word_seq_lens,
            batch_first=True,
            enforce_sorted=False
        )
        lstm_out, (_, _) = self.lstm(packed, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        out = self.linear(self.dropout(lstm_out))  # [batch_size, max_seq_length, vacab_size]

        return out
