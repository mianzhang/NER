import torch
import torch.nn as nn

PAD = '</pad>'
START = '<s>'
STOP = '</s>'
UNK = '</unk>'


def log_sum_exp_pytorch(t):
    max_score, _ = torch.max(t, dim=-1)
    return max_score + torch.log(torch.sum(torch.exp(t - max_score.unsqueeze(-1)), -1))


class CRF(nn.Module):

    def __init__(self, conf):
        super(CRF, self).__init__()
        self.device = conf.device
        self.tag_size = conf.dicts['tag_size']
        self.tag_to_idx = conf.dicts['tag_to_idx']
        self.PAD_IDX = self.tag_to_idx[PAD]
        self.START_IDX = self.tag_to_idx[START]
        self.STOP_IDX = self.tag_to_idx[STOP]

        self.trans = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        self.trans.data[self.START_IDX, :] = -10000
        self.trans.data[:, self.STOP_IDX] = -10000
        self.trans.data[:, self.PAD_IDX] = -10000
        self.trans.data[self.PAD_IDX, :] = -10000
        self.trans.data[self.PAD_IDX, self.STOP_IDX] = 0
        self.trans.data[self.PAD_IDX, self.PAD_IDX] = 0

    def _forward_alg(self, lstm_out, mask):
        max_seq_len = lstm_out.size(1)

        trans = self.trans.unsqueeze(0)  # [1 x C x C]
        alpha = lstm_out[:, 0, :] + trans[:, :, self.START_IDX]
        for t in range(1, max_seq_len):
            mask_t = mask[:, t].unsqueeze(1)
            emit_score = lstm_out[:, t, :].unsqueeze(2)  # [B x C x 1]
            alpha_t = alpha.unsqueeze(1) + emit_score + trans
            alpha_t = log_sum_exp_pytorch(alpha_t)  # [B x C]
            alpha = mask_t * alpha_t + (1 - mask_t) * alpha
        alpha = alpha + self.trans[self.STOP_IDX, :]  # [B]
        return torch.sum(log_sum_exp_pytorch(alpha))

    def _gold_score(self, lstm_out, tag_seq_tensor, word_seq_lens, mask):
        batch_size = lstm_out.size(0)
        max_seq_len = lstm_out.size(1)
        gold_score = torch.zeros(batch_size, dtype=torch.float).to(self.device)
        start_col = torch.full((batch_size, 1), self.START_IDX, dtype=torch.long).to(self.device)
        tag_seq_tensor = torch.cat([start_col, tag_seq_tensor], dim=1)
        for t in range(max_seq_len):
            mask_t = mask[:, t]
            next_tag = tag_seq_tensor[:, t + 1]  # [B]
            emit_t = lstm_out[:, t, :]  # [B x C]
            emit_t = torch.gather(emit_t, 1, next_tag.unsqueeze(1)).squeeze()  # [B]
            cur_tag = tag_seq_tensor[:, t]  # [B]
            trans_t = self.trans[next_tag, cur_tag]  # [B]
            gold_score = gold_score + (emit_t + trans_t) * mask_t
        last_tags = torch.gather(tag_seq_tensor, 1, word_seq_lens.unsqueeze(1)).squeeze()
        trans_to_end = self.trans[torch.LongTensor([self.STOP_IDX] * batch_size), last_tags]  # [B]
        gold_score += trans_to_end
        return torch.sum(gold_score)

    def neg_log_likelihood(self, lstm_out, data):
        word_seq_lens = data['word_seq_lens']
        tag_seq_tensor = data['tag_seq_tensor']
        mask = torch.gt(tag_seq_tensor, self.PAD_IDX).float()
        forward_score = self._forward_alg(lstm_out, mask)
        gold_score = self._gold_score(
            lstm_out,
            tag_seq_tensor,
            word_seq_lens,
            mask
        )
        return forward_score - gold_score

    def decode(self, lstm_out, mask):
        batch_size = lstm_out.size(0)
        max_seq_len = lstm_out.size(1)
        score = torch.full((batch_size, self.tag_size), -10000, dtype=torch.float).to(self.device)
        score[:, self.START_IDX] = 0
        bptrs = torch.zeros((batch_size, max_seq_len, self.tag_size), dtype=torch.long).to(self.device)

        trans = self.trans.unsqueeze(0)  # [1 x C x C]
        for t in range(max_seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # [B x 1]
            score_t = score.unsqueeze(1) + trans  # [B x C x C]
            max_scores, best_tag_ids = torch.max(score_t, dim=2)  # [B x C]
            max_scores += lstm_out[:, t, :]
            bptrs[:, t, :] = best_tag_ids
            score = mask_t * max_scores + (1 - mask_t) * score
        last_to_stop = self.trans[self.STOP_IDX, :].unsqueeze(0)  # [1 x C]
        score = score + last_to_stop

        best_scores, best_tags = torch.max(score, dim=1)  # [B]
        bptrs = bptrs.tolist()
        best_paths = []

        for i in range(batch_size):
            best_path = [best_tags[i].item()]
            bptr = bptrs[i]
            cur_seq_len = int(torch.sum(mask[i]).item())
            for j in reversed(range(1, cur_seq_len)):
                best_path.append(bptr[j][best_path[-1]])
            best_path.reverse()
            best_paths.append(best_path)

        return best_paths
