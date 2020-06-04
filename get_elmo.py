import sys
import argparse

import numpy as np
import torch
import pickle
from tqdm import tqdm
from allennlp.commands.elmo import ElmoEmbedder


def parse_sentence(elmo_vecs, mode="average"):
    """
    Load an ELMo embedder.
    :param elmo_vecs: the ELMo model results for a single sentence
    :param mode:
    :return:
    """
    if mode == "average":
        try:
            return np.average(elmo_vecs, 0)
        except:
            print(elmo_vecs.shape)
            exit(0)
    elif mode == 'weighted_average':
        return np.swapaxes(elmo_vecs, 0, 1)
    elif mode == 'last':
        return elmo_vecs[-1, :, :]
    elif mode == 'all':
        return elmo_vecs
    else:
        return elmo_vecs


def load_elmo(cuda_device: int):
    """
    Load a ElMo embedder
    :param cuda_device:
    :return:
    """
    return ElmoEmbedder(cuda_device=cuda_device)


def read_parse(elmo, samples, args):
    all_sents = [s.words for s in samples]
    all_vecs = []
    if args.batch_size == 1: # Not using batch
        for sent in tqdm(all_sents, desc="Elmo Embedding."):
            elmo_vecs = elmo.embed_sentence(sent)
            vec = torch.from_numpy(parse_sentence(elmo_vecs, mode=args.mode)).float()
            all_vecs.append(vec)
    else:
        for elmo_vecs in tqdm(elmo.embed_sentences(all_sents, batch_size=args.batch_size), desc="Elmo Embedding."):
            vec = torch.from_numpy(parse_sentence(elmo_vecs, mode=args.mode)).float()
            all_vecs.append(vec)

    return all_vecs

def main(args):

    elmo = load_elmo(args.device)

    # Load data.
    print("Loading data from '%s'." % args.data)
    data = torch.load(args.data)
    print("Loaded data.")

    ret = {}
    # Read train
    ret["train"] = read_parse(elmo, data["train"], args)
    # Read dev
    ret["dev"] = read_parse(elmo, data["dev"], args)
    # Read test
    ret["test"] = read_parse(elmo, data["test"], args)

    torch.save(ret, args.save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get_elmo.py")

    # data
    parser.add_argument("--data", required=True,
                        help="Path to data.")
    parser.add_argument("--save_data", required=True,
                        help="Path to save the data.")

    # inference config
    parser.add_argument("--device", type=int, default=3,
                        help="Computing device.")
    parser.add_argument("--mode", type=str, default="average",
                        help="Elmo mode.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch-based inference.")

    args = parser.parse_args()

    main(args)

