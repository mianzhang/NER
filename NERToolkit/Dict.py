import torch


class Dict:

    def __init__(self, data=None, lower=False):
        self.idx2label = {}
        self.label2idx = {}
        self.frequencies = {}
        self.lower = lower

        self.special = []

        if data is not None:
            for c in data:
                self.special.extend(data)

    def size(self):
        return len(self.idx2label)

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.label2idx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None):
        try:
            return self.idx2label[idx]
        except KeyError:
            return default

    def add(self, label):
        label = label.lower() if self.lower else label
        if label in self.label2idx:
            idx = self.label2idx[label]
        else:
            idx = len(self.idx2label)
            self.idx2label[idx] = label
            self.label2idx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size=None):
        if size and size >= self.size():
            return self

        if size is None:
            size = self.size()

        freq = torch.Tensor([self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        ret = Dict()
        ret.lower = self.lower

        for c in self.special:
            ret.add(c)

        for i in idx[:size]:
            ret.add(self.idx2label[i.item()])

        return ret

