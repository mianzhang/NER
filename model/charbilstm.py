import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Charbilstm(nn.Module):

    def __init__(self, conf):
        super(Charbilstm, self).__init__()
        print("Building bilstm character feature extractor.")
        self.char_embedding_dim = conf.char_embedding_dim
        self.char_hidden_dim = conf.char_hidden_dim
        self.char_to_idx = conf.dicts['char_to_idx']
        self.char_size = conf.dicts['char_size']
        self.embed_layer = nn.Embedding(
            self.char_size, self.char_embedding_dim)
        self.char_drop = nn.Dropout(p=0.5)
        self.char_lstm = nn.LSTM(
            self.char_embedding_dim,
            self.char_hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )

    def forward(self, data):
        char_seq_tensor = data['char_seq_tensor']
        char_seq_lens = data['char_seq_lens']
        max_word_len = char_seq_tensor.size(2)
        max_seq_len = char_seq_tensor.size(1)
        batch_size = char_seq_tensor.size(0)

        char_seq_tensor = char_seq_tensor.view(batch_size * max_seq_len, max_word_len)
        char_seq_lens = char_seq_lens.view(batch_size * max_seq_len)

        embedding = self.char_drop(self.embed_layer(char_seq_tensor))

        packed = pack_padded_sequence(
            embedding,
            char_seq_lens,
            batch_first=True,
            enforce_sorted=False
        )
        _, (lstm_out, _) = self.char_lstm(packed, None)  # [1*2 x B x H/2]
        out = torch.cat([lstm_out[0], lstm_out[1]], dim=1)  # [B x H]

        return out.view(batch_size, max_seq_len, self.char_hidden_dim)
