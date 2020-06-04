import torch
import torch.nn as nn
from .bilstm import Bilstm
from .crf import CRF
from .softmax import SoftmaxLayer

import NERToolkit

log = NERToolkit.utils.get_logger()

PAD = NERToolkit.Constants.PAD


class Tagger(nn.Module):

    def __init__(self, vocabs, args, word2vec=None):
        super(Tagger, self).__init__()
        self.device = args.device
        self.vocabs = vocabs
        self.tag_size = vocabs['tag'].size()
        self.bilstm = Bilstm(vocabs, args, word2vec)
        self.inference_layer = args.inference_layer
        if self.inference_layer == 'crf':
            self.crf_layer = CRF(vocabs, args)
        elif self.inference_layer == 'softmax':
            self.softmax_layer = SoftmaxLayer()

    def calculate_loss(self, data):
        lstm_out = self.bilstm(data)
        if self.inference_layer == 'crf':
            total_loss = self.crf_layer.neg_log_likelihood(lstm_out, data)
        else:
            total_loss = self.softmax_layer.neg_log_likelihood(lstm_out, data)
        return total_loss

    def forward(self, data):
        lstm_out = self.bilstm(data)
        word_seq_tensor = data['word_seq_tensor']
        PAD_IDX = self.vocabs['word'].lookup(PAD)
        mask = torch.gt(word_seq_tensor, PAD_IDX).float()
        if self.inference_layer == 'crf':
            best_paths = self.crf_layer.decode(lstm_out, mask)
        else:
            best_paths = self.softmax_layer(lstm_out, mask)
        return best_paths
