# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from languagemodeling import constants

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        for sent in sents:
            # Insert sentence markers.
            sent.insert(0,constants.BEGIN_SENTENCE_MARKER)
            sent.append(constants.END_SENTENCE_MARKER)
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                if ngram != (constants.BEGIN_SENTENCE_MARKER,):
                    counts[ngram] += 1
                    counts[ngram[:-1]] += 1


    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        assert isinstance(tokens,tuple)
        assert len(tokens) <= self.n

        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """


    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """