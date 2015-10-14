from collections import Counter
from math import log2

BEGIN_TAG = '<s>'
STOP_TAG = '</s>'

class HMM:

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self.tagset = tagset
        self.trans = trans
        self.out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self.tagset

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        if not prev_tags:
            prev_tags = ()

        probs = self.trans[prev_tags]
        if tag not in list(probs.keys()):
            return 0
        else:
            return probs[tag]

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        probs = self.out[tag]
        if word not in probs.keys():
            return 0
        else:
            return probs[word]

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        y = y + [STOP_TAG]
        prob = 1
        current_prev_tags = (BEGIN_TAG,) * (self.n - 1)

        for tag in y:
            prob *= self.trans_prob(tag, current_prev_tags)
            current_prev_tags = current_prev_tags[1:] + (tag,)
        return prob

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        prob = self.tag_prob(y)
        for i in range(len(x)):
            prob *= self.out_prob(x[i], y[i])
        return prob

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        prob = self.tag_prob(y)
        return log2(prob)

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        prob = self.prob(x, y)
        return log2(prob)

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
