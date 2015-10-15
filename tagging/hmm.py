from collections import defaultdict
from math import log2
import itertools

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

    def trans_log_prob(self, tag, prev_tags):
        """ Log Probability of a tag given prev_tags.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        prob = self.trans_prob(tag, prev_tags)
        return log2(prob) if (prob != 0) else float('-inf')

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

    def out_log_prob(self, word, tag):
        """Log Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        prob = self.out_prob(word, tag)
        return log2(prob) if (prob != 0) else float('-inf')

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
        return log2(prob) if (prob != 0) else float('-inf')

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
        tagger = ViterbiTagger(self)
        return tagger.tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        self._pi = defaultdict(lambda: defaultdict(tuple))
        # Initialize pi(0,*,*) = 1, backpointers = []
        self._pi[0][(BEGIN_TAG,) * (self.hmm.n - 1)] = (log2(1.0), [])

        N = len(sent)

        for k in range(1, N+1):
            for tag in self.hmm.tagset:
                prob_tag = (float('-inf'), '')
                for prev_tags, (prob, tag_sent) in self._pi[k-1].items():
                    # Note that this is iterate over all combinatios of tags Uk-(n-2) Uk, such that Ui belongs tu Ki(i) for each i.
                    # But, we just ignore those combinations that we know have pi(k-1,tag,Uk-(n-2),...,Uk-1) == 0
                    pik_1 = prob
                    q = self.hmm.trans_log_prob(tag, prev_tags)
                    e = self.hmm.out_log_prob(sent[k-1], tag)
                    prob = pik_1 + q + e
                    if prob > prob_tag[0]:
                        prob_tag = (prob, tag)
                if prob_tag[0] != float('-inf'):
                    self._pi[k][prev_tags[1:] + (tag,)] = (prob_tag[0], tag_sent + [tag])
        # Finally return the tag sequence whose last n-1 tags maximize trans_prob to STOP_TAG
        return max(self._pi[N].items(),
                   key=lambda x: x[1][0] * self.hmm.trans_log_prob(STOP_TAG, x[0])
                   )[1][1]
