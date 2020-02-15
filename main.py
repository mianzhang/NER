import os
import re
import random
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import argparse
from tqdm import tqdm
from model import Tagger
import logging
from datetime import datetime
import time

from utils.config import Config
from utils.functions import *


def set_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.random.manual_seed(conf.seed)
    if args.device.startswith('cuda'):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(conf.seed)
        torch.cuda.manual_seed_all(conf.seed)


parser = argparse.ArgumentParser(description='A simple NER toolkit.')
### Data parameters
parser.add_argument('--dataset', type=str,
    choices=['conll2003', 'ontonotes', 'msra', 'weibo'], default='conll2003')
parser.add_argument('--use_iobes', action='store_true')
### Trainning parameters
parser.add_argument('--from_begin', action='store_true')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda:3'], default='cpu')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--clip_value', type=float, default=5.0)
### Model parameters
parser.add_argument('--char_feature_extractor', type=str, default='none',
        choices=['none', 'cnn', 'bilstm'])
parser.add_argument('--inference_layer', type=str, default='crf',
        choices=['crf', 'softmax'])
parser.add_argument('--char_hidden_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=200)


def _logging():
    os.mkdir(logdir)
    logfile = os.path.join(logdir, 'log.log')
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def logging_config(args):
    logger.info('Config:')
    for k, v in vars(args).items():
        logger.info(k + ": " + str(v))
    logger.info("")


def evaluate(tagger, samples, conf):
    tagger.eval()
    idx_to_tag = conf.dicts['idx_to_tag']
    batchs = batching(samples, args)
    with torch.no_grad():
        tp, tp_fp, tp_fn = 0, 0, 0
        for batch in batchs:
            gold_paths = [sample.tag_ids for sample in batch]
            gold_paths = [
                [idx_to_tag[idx] for idx in path]
                for path in gold_paths
            ]
            data = padding(batch)
            for k, v in data.items():
                data[k] = v.to(conf.device)

            pred_paths = tagger(data)
            pred_paths = [
                [idx_to_tag[idx] for idx in path]
                for path in pred_paths
            ]
            if args.use_iobes:
                tp_, tp_fp_, tp_fn_ = cal_metrics_iobes(pred_paths, gold_paths)
            else:
                tp_, tp_fp_, tp_fn_ = cal_metrics(pred_paths, gold_paths)
            tp += tp_
            tp_fp += tp_fp_
            tp_fn += tp_fn_

        precision = 100.0 * tp / tp_fp if tp_fp != 0 else 0
        recall = 100.0 * tp / tp_fn if tp_fn != 0 else 0
        f_score = 2 * precision * recall / (precision + recall) \
            if precision != 0 or recall != 0 else 0
    return precision, recall, f_score


def main(conf):
    set_seed(conf)

    train_data = read_data(conf.train_file)
    dev_data = read_data(conf.dev_file)
    test_data = read_data(conf.test_file)

    if conf.use_iobes:
        map_to_iobes(conf.dataset, train_data)
        map_to_iobes(conf.dataset, dev_data)
        map_to_iobes(conf.dataset, test_data)

    conf.build_dicts(train_data, dev_data, test_data)

    conf.build_embedding_table()

    train_samples = conf.map_to_idx(train_data)
    print('number of samples:', len(train_samples))
    dev_samples = conf.map_to_idx(dev_data)
    test_samples = conf.map_to_idx(test_data)

    tagger = Tagger(conf).to(conf.device)

    opt = optim.SGD(tagger.parameters(),
                    lr=conf.learning_rate,
                    weight_decay=conf.weight_decay)

    random.shuffle(train_samples)
    batchs = batching(train_samples, conf)
    max_f_score = 0.0
    model_file = 'checkpoint-{}-{}-bilstm-{}.pt'.format(conf.dataset, conf.char_feature_extractor, conf.inference_layer)
    if not conf.from_begin:
        checkpoint = torch.load(model_file)
        tagger.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        max_f_score = checkpoint['max_f_score']

    for i in range(1, conf.epochs + 1):
        epoch_loss = 0
        start_time = time.time()
        for idx in np.random.permutation(len(batchs)):
            tagger.train()
            tagger.zero_grad()
            data = padding(batchs[idx])
            for k, v in data.items():
                data[k] = v.to(conf.device)

            nll = tagger.calculate_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            clip_grad_value_(tagger.parameters(), conf.clip_value)
            opt.step()
        end_time = time.time()
        logger.info(
            '[Epoch %d / %d] [Loss: %f] [Time: %f]' %
            (i, conf.epochs, epoch_loss, end_time - start_time)
        )

        dev_precision, dev_recall, dev_f_score = evaluate(tagger, dev_samples, conf)
        logger.info(
            '[Dev set] [precision %f] [recall %f] [fscore %f]' %
            (dev_precision, dev_recall, dev_f_score)
        )
        test_precision, test_recall, test_f_score = evaluate(tagger, test_samples, conf)
        logger.info(
            '[test set] [precision %f] [recall %f] [fscore %f]' %
            (test_precision, test_recall, test_f_score))
        logger.info("")

        if dev_f_score > max_f_score:
            max_f_score = dev_f_score
            torch.save({'model_state_dict': tagger.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'max_f_score': max_f_score,
                        'config': conf
                        }, model_file)
            logger.info('Save the best model.')


if __name__ == '__main__':
    args = parser.parse_args()
    global logdir
    logdir = '-'.join([
        'log/log',
        args.dataset,
        datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    ])
    _logging()
    logging_config(args)
    conf = Config(args, logger)
    main(conf)
