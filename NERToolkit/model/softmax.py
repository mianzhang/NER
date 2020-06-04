import torch
import torch.nn as nn


class SoftmaxLayer(nn.Module):

    def __init__(self):
        super(SoftmaxLayer, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')

    def forward(self, lstm_out, mask):
        batch_size = lstm_out.size(0)
        scores = self.log_softmax(lstm_out)
        _, best_tags = torch.max(scores, dim=-1)
        best_tags = best_tags.tolist()
        best_paths = []
        for i in range(batch_size):
            cur_seq_len = int(torch.sum(mask[i]).item())
            best_path = best_tags[i][0: cur_seq_len]
            best_paths.append(best_path)

        return best_paths

    def neg_log_likelihood(self, lstm_out, data):
        tag_seq_tensor = data['tag_seq_tensor']
        batch_size = tag_seq_tensor.size(0)
        max_seq_len = tag_seq_tensor.size(1)
        lstm_out = lstm_out.view(batch_size * max_seq_len, -1)
        scores = self.log_softmax(lstm_out)
        total_loss = self.loss_function(scores, tag_seq_tensor.view(batch_size * max_seq_len))

        return total_loss
