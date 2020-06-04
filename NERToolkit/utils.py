import sys
import logging
import random
import re

import numpy as np
import torch


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def read_data(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    words, tags = [], []
    for line in f:
        line = line.strip()
        if line == "":
            data.append((words, tags))
            words, tags = [], []
            continue
        line = line.split()
        if len(line) != 2:
            continue
        word = re.sub('\d', '0', line[0])
        words.append(word)
        tags.append(line[1])
    return data

def add_elmo(samples, all_vecs):
    for i in range(len(all_vecs)):
        samples[i].elmo_vec = all_vecs[i]

