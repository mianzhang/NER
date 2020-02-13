import torch
import torch.nn as nn
from .bilstm import Bilstm
from .crf import CRF
from .softmax import SoftmaxLayer

PAD = '</pad>'
START = '<s>'
STOP = '</s>'
UNK = '</unk>'


class Tagger(nn.Module):

    def __init__(self, conf):
        super(Tagger, self).__init__()
        self.device = conf.device
        self.tag_size = conf.dicts['tag_size']
        self.tag_to_idx = conf.dicts['tag_to_idx']
        self.PAD_IDX = self.tag_to_idx[PAD]
        self.START_IDX = self.tag_to_idx[START]
        self.STOP_IDX = self.tag_to_idx[STOP]
        self.bilstm = Bilstm(conf)
        self.inference_layer = conf.inference_layer
        if self.inference_layer == 'crf':
            self.crf_layer = CRF(conf)
        elif self.inference_layer == 'softmax':
            self.softmax_layer = SoftmaxLayer(conf)

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
        mask = torch.gt(word_seq_tensor, self.PAD_IDX).float()
        if self.inference_layer == 'crf':
            best_paths = self.crf_layer.decode(lstm_out, mask)
        else:
            best_paths = self.softmax_layer(lstm_out, mask)
        return best_paths
