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
        self.words_tags_counter = Counter(words_tags)
        self.most_freq_tag = Counter(tags).most_common(1)[0][0]
        self.words_mf_tags = {}
        for ((word, tag), count) in self.words_tags_counter.items():
            (w_tag, w_count) = self.words_mf_tags.get(word, ('', -1))
            if count > w_count:
                self.words_mf_tags[word] = (tag, count)

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        if self.unknown(w):
            return self.most_freq_tag
        else:
            return self.words_mf_tags[w][0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.words
