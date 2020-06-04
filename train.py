import argparse

import torch

import NERToolkit

parser = argparse.ArgumentParser(description="train.py")

# Data options
parser.add_argument("--corpus_name", required=True, type=str,
                    help="Corpus name.")
parser.add_argument("--data", required=True, type=str,
                    help="Data path.")
parser.add_argument("--elmo_data", type=str,
                    help="Path to elmo vecs.")

# Training parameters
parser.add_argument("--from_begin", action="store_true",
                    help="Training from begin.")
parser.add_argument("--device", type=str, default="cpu",
                    help="Computing device.")
parser.add_argument("--epochs", default=1, type=int,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Batch size.")
parser.add_argument("--optimizer", type=str, default="sgd",
                    choices=["sgd", "rmsprop", "adam"],
                    help="Name of optimizer.")
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="Learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-8,
                    help="Weight decay.")
parser.add_argument("--max_grad_value", default=5.0, type=float,
                    help="""If the norm of the gradient vector exceeds this,
                    normalize it to have the norm equal to max_grad_norm""")
parser.add_argument("--drop_rate", type=float, default=0.5,
                    help="Dropout rate.")

# Model parameters
parser.add_argument("--char_feature_extractor", type=str, default="none",
                    choices=["none", "cnn", "bilstm"],
                    help="Char feature extractor.")
parser.add_argument("--inference_layer", type=str, default="crf",
                    choices=["crf", "softmax"],
                    help="Inference layer")
parser.add_argument("--char_hidden_dim", type=int, default=50,
                    help="Character hidden dimension.")
parser.add_argument("--hidden_dim", type=int, default=200,
                    help="Hidden dimension.")
parser.add_argument("--embed_dim", required=True, type=int,
                    help="Dimension of word vecs.")
parser.add_argument("--char_embed_dim", type=int,
                    help="Dimension of char vecs.")
parser.add_argument("--context_emb", type=str, default="none", choices=["none", "elmo"],
                    help="Context embeddings.")

# Pretrained word vectors
parser.add_argument("--word2vec", default="", type=str,
                    help="Pretrained word vectors.")
# Other parameters
parser.add_argument("--seed", default=24, type=int,
                    help="Random seed.")

args = parser.parse_args()
NERToolkit.utils.set_seed(args.seed)

log = NERToolkit.utils.get_logger()
log.debug(args)


def main(args):
    # Load data.
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data)
    vocabs = data["vocabs"]
    word2vec = torch.load(args.word2vec)
    log.info("Loaded data.")

    for split in ('train', 'dev', 'test'):
        for sample in data[split]:
            sample.map_to_id(vocabs)

    # Elmo
    if args.context_emb == "elmo":
        elmo_data = torch.load(args.elmo_data)
        for split in ('train', 'dev', 'test'):
            NERToolkit.utils.add_elmo(data[split], elmo_data[split])

    log.debug(vocabs["char"].label2idx)

    trainset = NERToolkit.Dataset(data['train'], args.batch_size, args)
    devset = NERToolkit.Dataset(data['dev'], args.batch_size, args)
    testset = NERToolkit.Dataset(data['test'], args.batch_size, args)
    log.info("Loaded datasets.")

    # Build model.
    log.debug("Building model...")
    model = NERToolkit.Tagger(vocabs, args, word2vec).to(args.device)
    for name, value in model.named_parameters():
        print("name: {}\t grad: {}".format(name, value.requires_grad))
    opt = NERToolkit.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(filter(lambda p: p.requires_grad, model.parameters()), args.optimizer)

    model_file = "save/ckpt-{}-{}-bilstm-{}-{}.pt".format(
        args.corpus_name, args.char_feature_extractor, args.inference_layer, args.context_emb)

    nParams = sum([p.nelement() for p in model.parameters()])
    log.debug("* number of parameters: %d" % nParams)

    coach = NERToolkit.Coach(model, vocabs, trainset, devset, testset, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
