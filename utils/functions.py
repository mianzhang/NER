import re
from tqdm import tqdm
import torch


def read_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in tqdm(f.readlines()):
            line = line.rstrip()
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


def batching(samples, conf):
    batch_size = conf.batch_size
    batched_samples = []
    if len(samples) % batch_size == 0:
        batch_num = len(samples) // batch_size
    else:
        batch_num = len(samples) // batch_size + 1

    for i in range(batch_num):
        batched_samples.append(
            samples[i * batch_size: (i + 1) * batch_size]
        )

    return batched_samples

def padding(samples, predict=False):
    batch_size = len(samples)
    word_seq_lens = torch.tensor(
        list(map(lambda sap: len(sap.word_ids), samples)),
        dtype=torch.long
    )
    max_seq_len = torch.max(word_seq_lens)
    char_seq_lens = torch.ones((batch_size, max_seq_len), dtype=torch.long)
    for i, sample in enumerate(samples):
        for j, char_seq in enumerate(sample.char_ids):
            char_seq_lens[i][j] = len(char_seq)
    max_word_len = torch.max(char_seq_lens)

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    if not predict:
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    else:
        tag_seq_tensor = None
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long)

    for i, sample in enumerate(samples):
        word_seq_tensor[i, 0: word_seq_lens[i]] = torch.tensor(
            sample.word_ids,
            dtype=torch.long
        )
        if not predict:
            tag_seq_tensor[i, 0: word_seq_lens[i]] = torch.tensor(
                sample.tag_ids,
                dtype=torch.long
            )
        for j, char_id in enumerate(sample.char_ids):
            char_seq_tensor[i, j, 0: char_seq_lens[i][j]] = torch.tensor(
                char_id,
                dtype=torch.long
            )

    return dict(
        word_seq_tensor=word_seq_tensor,
        tag_seq_tensor=tag_seq_tensor,
        char_seq_tensor=char_seq_tensor,
        word_seq_lens=word_seq_lens,
        char_seq_lens=char_seq_lens,
    )


def cal_metrics(pred_paths, gold_paths):

    def get_name_entities(path):
        length = len(path)
        name_entitis = set()
        i = 0
        while i < length:
            cur_tag = path[i]
            if cur_tag.startswith('B-'):
                tag = cur_tag[2:]
                in_tag = cur_tag.replace('B', 'I')
                j = i + 1
                while j < length and path[j] == in_tag:
                    j += 1
                name_entitis.add(tag + "-" + str(i) + "-" + str(j))
                i = j
            else:
                i += 1
        return name_entitis

    tp, tp_fp, tp_fn = 0, 0, 0
    for pred_path, gold_path in zip(pred_paths, gold_paths):
        pred_nes = get_name_entities(pred_path)
        gold_nes = get_name_entities(gold_path)
        tp += len(pred_nes.intersection(gold_nes))
        tp_fp += len(pred_nes)
        tp_fn += len(gold_nes)

    return tp, tp_fp, tp_fn


def cal_metrics_iobes(pred_paths, gold_paths):

    def get_name_entities(path):
        name_entitis = set()
        start = -1
        for i in range(len(path)):
            cur_tag = path[i]
            if cur_tag.startswith('B'):
                start = i
            if cur_tag.startswith('E'):
                name_entitis.add(str(start) + "-" + str(i) + "-" + cur_tag[2:])
            if cur_tag.startswith('S'):
                name_entitis.add(str(i) + "-" + str(i) + "-" + cur_tag)
        return name_entitis

    tp, tp_fp, tp_fn = 0, 0, 0
    for pred_path, gold_path in zip(pred_paths, gold_paths):
        pred_nes = get_name_entities(pred_path)
        gold_nes = get_name_entities(gold_path)
        tp += len(pred_nes.intersection(gold_nes))
        tp_fp += len(pred_nes)
        tp_fn += len(gold_nes)

    return tp, tp_fp, tp_fn


def map_to_iobes(dataset, data):
    if dataset in ['conll2003', 'weibo']:
        for words, tags in data:
            for i in range(len(tags)):
                cur_tag = tags[i]
                if i == len(tags) - 1:
                    if cur_tag.startswith('B-'):
                        tags[i] = cur_tag.replace('B-', 'S-')
                    elif cur_tag.startswith('I'):
                        tags[i] = cur_tag.replace('I-', 'E-')
                else:
                    next_tag = tags[i + 1]
                    if cur_tag.startswith('B-'):
                        if next_tag.startswith('O') or next_tag.startswith('B-'):
                            tags[i] = cur_tag.replace('B-', 'S-')
                    elif cur_tag.startswith('I'):
                        if next_tag.startswith('O') or next_tag.startswith('B-'):
                            tags[i] = cur_tag.replace('I-', 'E-')
    elif dataset == 'ontonotes':
        for words, tags in data:
            for i in range(len(tags)):
                cur_tag = tags[i]
                if cur_tag.startswith('M-'):
                    tags[i] = cur_tag.replace('M-', 'I-')

