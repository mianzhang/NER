import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .charbilstm import Charbilstm
from .charcnn import CharCNN
import NERToolkit

log = NERToolkit.utils.get_logger()


class Bilstm(nn.Module):

    def __init__(self, vocabs, args, word2vec=None):
        super(Bilstm, self).__init__()
        self.device = args.device
        self.vacab_size = vocabs['word'].size()
        self.tag_size = vocabs['tag'].size()
        self.embed_dim = args.embed_dim
        input_size = self.embed_dim
        self.char_feature_extractor = args.char_feature_extractor
        self.hidden_dim = args.hidden_dim
        if self.char_feature_extractor != 'none':
            input_size += args.char_hidden_dim
            if self.char_feature_extractor == 'cnn':
                self.charcnn = CharCNN(vocabs, args)
            elif self.char_feature_extractor == 'bilstm':
                self.charbilstm = Charbilstm(vocabs, args)
        self.context_emb = args.context_emb
        if self.context_emb == "elmo":
            input_size += 1024
        self.embed_layer = self.build_embed_layer(word2vec)
        self.word_drop = nn.Dropout(p=args.drop_rate)
        self.lstm = nn.LSTM(
            input_size,
            self.hidden_dim // 2,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.linear = nn.Linear(self.hidden_dim, self.tag_size)

    def build_embed_layer(self, embedding_table):
        if embedding_table is not None:
            log.info("Built pretrained word2vec.")
            embed_layer = nn.Embedding.from_pretrained(
                embedding_table,
                freeze=False)
        else:
            embed_layer = nn.Embedding(self.vacab_size, self.char_embedding_dim)
        log.info(embed_layer)
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
        if self.context_emb == "elmo":
            elmo_tensor = data["elmo_tensor"]  # [B x L x H]
            embedding = torch.cat([embedding, elmo_tensor], dim=2)

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
