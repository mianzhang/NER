import time
import copy

import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_value_

import NERToolkit

log = NERToolkit.utils.get_logger()


class Coach:

    def __init__(self, model, vocabs, trainset, devset, testset, opt, args):
        self.model = model
        self.vocabs = vocabs
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.opt = opt
        self.args = args
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        # Train
        self.trainset.shuffle()
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)

            dev_precision, dev_recall, dev_f1 = self.evaluate()
            log.info("[Dev set] [precision %f] [recall %f] [f1 %f]" %
                     (dev_precision, dev_recall, dev_f1))
            test_precision, test_recall, test_f1 = self.evaluate(test=True)
            log.info("[test set] [precision %f] [recall %f] [f1 %f]" %
                     (test_precision, test_recall, test_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

        # The best
        self.model.load_state_dict(best_state)
        log.info("Best in epoch {}:".format(best_epoch))
        dev_precision, dev_recall, dev_f1 = self.evaluate()
        log.info("[Dev set] [precision %f] [recall %f] [f1 %f]" %
                 (dev_precision, dev_recall, dev_f1))
        test_precision, test_recall, test_f1 = self.evaluate(test=True)
        log.info("[test set] [precision %f] [recall %f] [f1 %f]" %
                 (test_precision, test_recall, test_f1))

        return best_dev_f1, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        for idx in tqdm(np.random.permutation(len(self.trainset)), desc="train one epoch"):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to(self.args.device)

            nll = self.model.calculate_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            tp, tp_fp, tp_fn = 0, 0, 0
            for idx in range(len(dataset)):
                gold_paths = [sample.tags for sample in dataset.raw_batch(idx)]
                data = dataset[idx]
                for k, v in data.items():
                    data[k] = v.to(self.args.device)

                pred_paths = self.model(data)
                pred_paths = [
                    [self.vocabs["tag"].get_label(j) for j in path] for path in pred_paths]
                tp_, tp_fp_, tp_fn_ = cal_metrics_bieos(pred_paths, gold_paths)
                tp += tp_
                tp_fp += tp_fp_
                tp_fn += tp_fn_

            precision = 100.0 * tp / tp_fp if tp_fp != 0 else 0
            recall = 100.0 * tp / tp_fn if tp_fn != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) \
                if precision != 0 or recall != 0 else 0
        return precision, recall, f1


def cal_metrics_bieos(pred_paths, gold_paths):

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
