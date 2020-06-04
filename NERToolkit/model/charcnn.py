import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):

    def __init__(self, vocabs, args):
        super(CharCNN, self).__init__()
        print("Building cnn character feature extractor.")
        self.char_size = vocabs['char'].size()
        self.char_embed_dim = args.char_embed_dim
        self.char_hidden_dim = args.char_hidden_dim
        self.embed_layer = nn.Embedding(self.char_size, self.char_embed_dim)
        self.char_drop = nn.Dropout(p=0.5)
        self.char_cnn = nn.Conv1d(self.char_embed_dim, self.char_hidden_dim, 3, padding=1)

    def forward(self, data):
        char_seq_tensor = data['char_seq_tensor']
        char_seq_lens = data['char_seq_lens']
        max_word_len = char_seq_tensor.size(2)
        max_seq_len = char_seq_tensor.size(1)
        batch_size = char_seq_tensor.size(0)

        char_seq_tensor = char_seq_tensor.view(batch_size * max_seq_len, max_word_len)
        char_seq_lens = char_seq_lens.view(batch_size * max_seq_len)

        char_embeds = self.char_drop(self.embed_layer(char_seq_tensor))  # [BxL, W, E]
        char_embeds = char_embeds.transpose(1, 2).contiguous()  # [BxL, E, W]
        hiddens = self.char_cnn(char_embeds)  # [BxL, H, W]
        char_rep = F.max_pool1d(hiddens, max_word_len)  # [BxL, H, 1]
        char_rep = char_rep.view(batch_size, max_seq_len, -1).contiguous()

        return char_rep

