import argparse

from tqdm import tqdm


def clean_word2vec(args):
    f = open(args.word2vec)
    f_err = open(args.word2vec + ".err", 'w')
    f_clean = open(args.word2vec + ".clean", 'w')

    for entry in tqdm(f.readlines(), desc="clean word2vec"):
        line = entry.strip()
        fields = line.split()
        token = fields[0]
        try:
            vec = list(map(float, fields[1:]))
        except ValueError:
            f_err.write(entry)
            continue
        if len(vec) != args.embed_dim:
            f_err.write(entry)
            continue
        f_clean.write(entry)
    f.close()
    f_err.close()
    f_clean.close()


def main(args):
    clean_word2vec(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="clean.py")

    # Input data
    parser.add_argument("--word2vec", default="", type=str,
                        help="Path to pretrained word vectors.")
    parser.add_argument("--embed_dim", required=True, type=int,
                        help="Dimension of word vecs.")

    args = parser.parse_args()

    main(args)
