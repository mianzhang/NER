import math
import random

import torch


class Dataset:

    def __init__(self, samples, batch_size, args, gold=True):
        self.samples = samples
        self.batch_size = batch_size
        self.args = args
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.gold = gold

    def __len__(self):
        return self.num_batches

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        # batch = sorted(batch, key=lambda s: len(s.words), reverse=True)

        return batch

    def shuffle(self):
        random.shuffle(self.samples)

    def padding(self, samples):
        batch_size = len(samples)
        word_seq_lens = torch.tensor(
            list(map(lambda sap: len(sap.word_ids), samples)),
            dtype=torch.long)
        max_seq_len = torch.max(word_seq_lens)
        char_seq_lens = torch.ones((batch_size, max_seq_len), dtype=torch.long)
        for i, sample in enumerate(samples):
            for j, char_seq in enumerate(sample.char_ids):
                char_seq_lens[i][j] = len(char_seq)
        max_word_len = torch.max(char_seq_lens)

        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        if self.gold:
            tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        else:
            tag_seq_tensor = None
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long)

        if self.args.context_emb == "elmo":
            elmo_tensor = torch.zeros((batch_size, max_seq_len, 1024))
        else:
            elmo_tensor = None

        for i, sample in enumerate(samples):
            word_seq_tensor[i, 0: word_seq_lens[i]] = torch.tensor(
                sample.word_ids,
                dtype=torch.long
            )
            if self.gold:
                tag_seq_tensor[i, 0: word_seq_lens[i]] = torch.tensor(
                    sample.tag_ids, dtype=torch.long)
            for j, char_id in enumerate(sample.char_ids):
                char_seq_tensor[i, j, 0: char_seq_lens[i][j]] = torch.tensor(
                    char_id, dtype=torch.long)

            if self.args.context_emb == "elmo":
                elmo_tensor[i, 0: word_seq_lens[i], :] = sample.elmo_vec

        ret = dict(
            word_seq_tensor=word_seq_tensor,
            tag_seq_tensor=tag_seq_tensor,
            char_seq_tensor=char_seq_tensor,
            word_seq_lens=word_seq_lens,
            char_seq_lens=char_seq_lens,
        )

        if self.args.context_emb == "elmo":
            ret["elmo_tensor"] = elmo_tensor

        return ret

    def __getitem__(self, index):
        batch = self.raw_batch(index)

        return self.padding(batch)


