import argparse
from tqdm import tqdm

import NERToolkit


def bieos(file):
    data = NERToolkit.utils.read_data(file)
    f = open(file + ".bieos", 'w')
    for tokens, tags in tqdm(data, desc="convert to bieos."):
        n = len(tokens)
        i, j = 0, 0
        while i < n:
            tag = tags[i]
            if tag.startswith("B-"):
                suffix = tag.strip("B-")
                j = i + 1
                while j < n and not tags[j].startswith(("B-", "S-", "O")):
                    j += 1
                x = len(tags[i: j])
                if x == 1:
                    tags[i] = "S-" + suffix
                else:
                    for k in range(i + 1, j - 1):
                        tags[k] = "I-" + suffix
                    tags[j - 1] = "E-" + suffix
                i = j
            else:
                i += 1
        for token, tag in zip(tokens, tags):
            f.write(token + ' ' + tag + '\n')
        f.write('\n')
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    # Input data
    parser.add_argument("--train", required=True,
                        help="Path to the training data.")
    parser.add_argument("--dev", required=True,
                        help="Path to the dev data.")
    parser.add_argument("--test", required=True,
                        help="Path to the test data.")

    args = parser.parse_args()
    bieos(args.train)
    bieos(args.dev)
    bieos(args.test)

