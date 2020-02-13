import os
from tqdm import tqdm
import numpy as np
import pickle
import torch

from .sample import Sample

PAD = '</pad>'
START = '<s>'
STOP = '</s>'
UNK = '</unk>'


class Config:

    def __init__(self, args, logger):
        self.logger = logger

        ### Data parameters
        self.dataset = args.dataset
        if self.dataset == 'conll2003':
            self.embedding_file = './embeddings/glove.6B.100d.txt'
            self.embedding_dim = 100
            self.char_embedding_dim = 25
        elif self.dataset in ['ontonotes', 'msra']:
            self.embedding_file = './embeddings/sgns.wiki.bigram-char'
            self.embedding_dim = 300
            self.char_embedding_dim = 75
        elif self.dataset in ['weibo']:
            self.embedding_file = './embeddings/sgns.weibo.word'
            self.embedding_dim = 300
            self.char_embedding_dim = 75
        self.train_file = '/'.join(['data', self.dataset, 'train.txt'])
        self.dev_file = '/'.join(['data', self.dataset, 'dev.txt'])
        self.test_file = '/'.join(['data', self.dataset, 'test.txt'])
        self.use_iobes = args.use_iobes
        self.dicts = None
        self.embedding_dict = self._build_embedding_dict()
        self.embedding_table = None

        # Training parameters
        self.from_begin = args.from_begin
        self.device = args.device
        self.seed = 0
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.clip_value = args.clip_value

        ### Model parameters
        self.char_feature_extractor = args.char_feature_extractor
        self.inference_layer = args.inference_layer
        self.char_hidden_dim = args.char_hidden_dim
        self.hidden_dim = args.hidden_dim

    def build_dicts(self, train_data, dev_data, test_data):
        word_to_idx = {PAD: 0, UNK: 1}
        tag_to_idx = {PAD: 0}
        idx_to_word = [PAD, UNK]
        idx_to_tag = [PAD]
        char_to_idx = {PAD: 0, UNK: 1}
        idx_to_char = [PAD, UNK]

        if dev_data is not None:
            for words, _ in train_data + dev_data + test_data:
                for word in words:
                    if word not in word_to_idx:
                        word_to_idx[word] = len(word_to_idx)
                        idx_to_word.append(word)
        else:
            for words, _ in train_data + test_data:
                for word in words:
                    if word not in word_to_idx:
                        word_to_idx[word] = len(word_to_idx)
                        idx_to_word.append(word)

        for words, tags in train_data:
            for tag in tags:
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = len(tag_to_idx)
                    idx_to_tag.append(tag)

            for word in words:
                for c in word:
                    if c not in char_to_idx:
                        char_to_idx[c] = len(char_to_idx)
                        idx_to_char.append(c)

        tag_to_idx[START] = len(tag_to_idx)
        tag_to_idx[STOP] = len(tag_to_idx)
        idx_to_tag.append(START)
        idx_to_tag.append(STOP)

        self.logger.info('num of words: %d: ' % len(idx_to_word))
        self.logger.info('num of chars: %d: ' % len(idx_to_char))
        print('tags:', idx_to_tag)
        self.dicts = dict(
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            tag_to_idx=tag_to_idx,
            idx_to_tag=idx_to_tag,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            vocab_size=len(idx_to_word),
            tag_size=len(idx_to_tag),
            char_size=len(idx_to_char)
        )

    def _build_embedding_dict(self):
        embedding_dict_file = self.embedding_file + '.pkl'
        if os.path.exists(embedding_dict_file):
            with open(embedding_dict_file, 'rb') as f:
                embedding_dict = pickle.load(f)
            return embedding_dict

        embedding_dict = {}
        count = 0
        with open(self.embedding_file, 'r') as f:
            for line in tqdm(f.readlines()):
                count += 1
                line = line.rstrip()
                if line == "":
                    continue
                line = line.split()
                word = line[0]
                try:
                    vec = list(map(float, line[1:]))
                except ValueError:
                    print(count, word)
                    continue
                if len(vec) != self.embedding_dim:
                    print(count, word, len(vec))
                    continue

                embedding_dict[word] = np.array(vec).reshape(1, self.embedding_dim)
        with open(embedding_dict_file, 'wb') as f:
            pickle.dump(embedding_dict, f)

        self.embedding_dict = embedding_dict

    def build_embedding_table(self):
        word_to_idx = self.dicts['word_to_idx']
        idx_to_word = self.dicts['idx_to_word']

        scale = np.sqrt(3.0 / self.embedding_dim)
        vacab_size = len(word_to_idx)
        embedding_table = np.zeros((vacab_size, self.embedding_dim))
        count = 0
        for i in range(vacab_size):
            word = idx_to_word[i]
            if word.lower() in self.embedding_dict:
                count += 1
                embedding_table[i] = self.embedding_dict[word.lower()]
            else:
                embedding_table[i] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
        self.logger.info('[%d / %d] words has corresponding pre-trained embeddings.' % (count, vacab_size))
        self.logger.info('number of pre-trained embeddings: %d' % len(self.embedding_dict))

        self.embedding_table = embedding_table

    def map_to_idx(self, data, predict=False):
        samples = []
        word_to_idx = self.dicts['word_to_idx']
        tag_to_idx = self.dicts['tag_to_idx']
        char_to_idx = self.dicts['char_to_idx']

        if not predict:
            for words, tags in data:
                word_ids = [word_to_idx[word] if word in word_to_idx else word_to_idx[UNK]
                            for word in words]
                tag_ids = [tag_to_idx[tag] for tag in tags]
                char_ids = []
                for word in words:
                    char_id = [
                        char_to_idx[c] if c in char_to_idx else char_to_idx[UNK]
                        for c in word
                    ]
                    char_ids.append(char_id)
                samples.append(Sample(word_ids, tag_ids, char_ids))
        else:
            for words in data:
                word_ids = [word_to_idx[word] if word in word_to_idx else word_to_idx[UNK]
                            for word in words]
                char_ids = []
                for word in words:
                    char_id = [
                        char_to_idx[c] if c in char_to_idx else char_to_idx[UNK]
                        for c in word
                    ]
                    char_ids.append(char_id)
                samples.append(Sample(word_ids, None, char_ids))

        return samples

