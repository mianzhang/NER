import NERToolkit


UNK = NERToolkit.Constants.UNK


class Sample:

    def __init__(self, words, tags, chars):
        self.words = words
        self.tags = tags
        self.chars = chars
        self.word_ids = None
        self.tag_ids = None
        self.char_ids = None
        self.elmo_vec = None

    def map_to_id(self, vocabs):
        UNK_IDX = vocabs["word"].lookup(UNK)
        self.word_ids = [vocabs['word'].lookup(w, UNK_IDX) for w in self.words]
        self.tag_ids = [vocabs['tag'].lookup(t) for t in self.tags]
        UNK_IDX = vocabs["char"].lookup(UNK)
        self.char_ids = []
        for word in self.words:
                self.char_ids.append([vocabs['char'].lookup(c, UNK_IDX) for c in word])

    def __repr__(self):
        ret = "\n"
        ret += "words: "
        for w in self.words:
                ret = ret + w + ' '
        ret += "\ntags: "
        for t in self.tags:
                ret = ret + t + ' '
        ret += "\nword_ids: "
        for idx in self.word_ids:
                ret = ret + str(idx) + ' '
        ret += "\ntag_ids: "
        for idx in self.tag_ids:
                ret = ret + str(idx) + ' '

        return ret

