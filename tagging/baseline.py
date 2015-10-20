import itertools
from collections import Counter


class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        words_tags = list(itertools.chain(*tagged_sents))
        words, tags = zip(*words_tags)

        self.words = words
        self.freq_words_tags = Counter(words_tags)
        self.most_freq_tag = Counter(tags).most_common(1)[0][0]
        self.most_freq_tags = {}
        word_set = set(words)
        for word in word_set:
            self.most_freq_tags[word] = self.get_tag_word(word)

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        return self.most_freq_tags[w]

    def get_tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            tag = self.most_freq_tag
        else:
            most_common_word_tag_count = max([fwt for fwt in list(self.freq_words_tags.items())
                                              if fwt[0][0] == w],
                                             key=lambda x: x[1])
            tag = most_common_word_tag_count[0][1]
        return tag

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.words
