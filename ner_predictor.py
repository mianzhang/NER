import re
import torch
from model.tagger import Tagger


class NerPredictor:

    def __init__(self):
        self.device = torch.device('cpu')
        model_file = 'checkpoint-ontonotes.pt'
        checkpoint = torch.load(model_file, map_location=self.device)
        self.conf = checkpoint['config']
        self.conf.device = self.device
        self.tagger = Tagger(self.conf)
        self.tagger.load_state_dict(checkpoint['model_state_dict'])

    def text_to_samples(self, text):
        sentences = re.split('(。|！|\!|\.|？|\?|；|;|～|~)', text)  # 保留分割符

        sents = []
        charseqs = []
        for i in range(int(len(sentences) / 2)):
            sent = sentences[2 * i] + sentences[2 * i + 1]
            charseq = [c for c in sent]
            sents.append(sent)
            charseqs.append(charseq)

        samples = self.conf.map_to_idx(charseqs, predict=True)
        return samples, charseqs, sents

    def decode(self, text):

        def get_name_entities(path, sent):
            name_entities = []
            start = -1
            for i in range(len(path)):
                cur_tag = path[i]
                if cur_tag.startswith('B'):
                    start = i
                if cur_tag.startswith('E'):
                    name_entities.append(sent[start: i + 1])
                if cur_tag.startswith('S'):
                    name_entities.append(sent[i: i + 1])
            return name_entities

        idx_to_tag = self.conf.dicts['idx_to_tag']
        samples, charseqs, sents = self.text_to_samples(text)
        with torch.no_grad():
            self.tagger.eval()
            data = self.conf.padding(samples, predict=True)
            pred_paths = self.tagger(data)
            pred_paths = [
                [idx_to_tag[idx] for idx in path]
                for path in pred_paths
            ]
            nes_sents = []
            for path, sent in zip(pred_paths, sents):
                nes = get_name_entities(path, sent)
                nes_sents.append(nes)

        self.write_res(sents, nes_sents)

        return nes_sents, sents

    def write_res(self, sents, nes_sents):
        with open('out.txt', 'w', encoding='utf-8') as f:
            for sent, nes_sent in zip(sents, nes_sents):
                f.write(sent)
                f.write('\n')
                f.write('\t'.join([ne for ne in nes_sent]))
                f.write('\n')


def predict(text):
    print("Extracting...")
    ner_predictor = NerPredictor()
    nes_sents, sents = ner_predictor.decode(text)
    print(sents)
    print(nes_sents)
    print("Finished.")

    return nes_sents, sents


if __name__ == '__main__':
    s = "四川人在东京吃火锅。但是姚明更喜欢在上海吃火锅！"
    predict(s)
