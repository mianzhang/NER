import argparse
from tqdm import tqdm
import torch
import numpy as np

import NERToolkit


log = NERToolkit.utils.get_logger()


def make_vocabs(all_data, args):
    word_vocab = NERToolkit.Dict(
        [NERToolkit.Constants.PAD, NERToolkit.Constants.UNK])
    tag_vocab = NERToolkit.Dict(
        [NERToolkit.Constants.PAD, NERToolkit.Constants.BOS, NERToolkit.Constants.EOS])
    char_vocab = NERToolkit.Dict(
        [NERToolkit.Constants.PAD, NERToolkit.Constants.UNK])

    for words, tags in tqdm(all_data, desc="make_vocabs"):
        for word in words:
            word = word
            word_vocab.add(word)
            for c in word:
                char_vocab.add(c)
        for tag in tags:
            tag_vocab.add(tag)
    word_vocab = word_vocab.prune()
    tag_vocab = tag_vocab.prune()
    char_vocab = char_vocab.prune()

    log.info("Created vocabs:\n\t#word: %d\n\t#tag: %d\n\t#char: %d"
             % (word_vocab.size(), tag_vocab.size(), char_vocab.size()))

    return {"word": word_vocab, "tag": tag_vocab, "char": char_vocab}


def make_word2vec(filepath, embed_dim, vocab):
    word2vec = {}
    log.info("Start loading pretrained word vecs")
    for line in tqdm(open(filepath).readlines(), desc="make word2vec"):
        line = line.strip()
        fields = line.split()
        word = fields[0]
        vec = list(map(float, fields[1:]))
        word2vec[word] = torch.Tensor(vec)

    ret = []
    count = 0
    scale = np.sqrt(3.0 / embed_dim)
    for idx in range(vocab.size()):
        word = vocab.idx2label[idx]
        if word in word2vec:
            count += 1
            ret.append(word2vec[word])
        elif word.lower() in word2vec:
            ret.append(word2vec[word.lower()])
            count += 1
        else:
            ret.append(torch.Tensor(embed_dim).uniform_(-scale, scale))

    ret = torch.stack(ret)
    log.info('[%d / %d] words has corresponding pre-trained embeddings.' % (count, vocab.size()))

    return ret


def make_samples(data):
    ret = []
    for words, tags in tqdm(data, desc="make_samples"):
        chars = []
        for word in words:
            t = [c for c in word]
            chars.append(t)
        ret.append(NERToolkit.Sample(words, tags, chars))

    return ret


def main(args):

    train_data = NERToolkit.utils.read_data(args.train)
    dev_data = NERToolkit.utils.read_data(args.dev)
    test_data = NERToolkit.utils.read_data(args.test)

    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(train_data + dev_data + test_data, args)

    log.info("Preparing train samples...")
    train_samples = make_samples(train_data)
    log.info("Preparing dev samples...")
    dev_samples = make_samples(dev_data)
    log.info("Preparing test samples...")
    test_samples = make_samples(test_data)

    log.info("Preparing pretrained word vectors...")
    word2vec = make_word2vec(args.word2vec, args.embed_dim, vocabs["word"])
    log.info("Saving pretrained word vectors to '%s'..." % (args.save_data + ".word2vec"))
    torch.save(word2vec, args.save_data + ".word2vec")

    log.info("Saving data to '%s'..." % (args.save_data + ".data.pt"))
    save_data = {"vocabs": vocabs, "train": train_samples, "dev": dev_samples, "test": test_samples}
    torch.save(save_data, args.save_data + ".data.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess.py")

    # Input data
    parser.add_argument("--train", required=True,
                        help="Path to the training data.")
    parser.add_argument("--dev", required=True,
                        help="Path to the dev data.")
    parser.add_argument("--test", required=True,
                        help="Path to the test data.")
    parser.add_argument("--word2vec", default="", type=str,
                        help="Path to pretrained word vectors.")
    parser.add_argument("--embed_dim", type=int,
                        help="Dimension of word vecs.")

    # Ops
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle data.")
    parser.add_argument('--seed', type=int, default=24,
                        help="Random seed")
    parser.add_argument('--lower', action='store_true',
                        help='lowercase data')

    # Output data
    parser.add_argument("--save_data", required=True,
                        help="Path to the output data.")

    args = parser.parse_args()

    NERToolkit.utils.set_seed(args.seed)

    main(args)
