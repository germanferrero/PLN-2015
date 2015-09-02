# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from languagemodeling import constants
from math import log

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        begin_sentence_list = [constants.BEGIN_SENTENCE_MARKER] * (n-1)
        for sent in sents:
            # Insert sentence markers.
            sent = begin_sentence_list + sent
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
        # assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        print (token)
        print (tokens)
        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        prob = 1
        my_sent = ([constants.BEGIN_SENTENCE_MARKER] * (self.n - 1)) + sent
        my_sent.append(constants.END_SENTENCE_MARKER)
        for i in range(self.n-1,len(my_sent)):
            token = my_sent[i]
            prob *= self.cond_prob(my_sent[i],my_sent[i-(self.n-1):i])
            if prob == 0:
                print (prob)
                break
        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        prob = self.sent_prob(sent)
        if prob == 0:
            return float('-inf')
        return log(self.sent_prob(sent),2)

class NGramGenerator:

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.model = model
        # Initialize probs, a dict of float's dicts.
        self.probs = defaultdict(lambda: defaultdict(float))
        self.sorted_probs = defaultdict(list)
        # First count tokens appearances.
        for k,v in self.model.counts.items():
            if len(k) == (self.model.n):
                self.probs[k[:-1]][k[-1]] = v

        # Then calculate probability distribution from total appearences.
        for key,value_dict in self.probs.items():
            total_sum = sum(value_dict.values())
            for sub_key in value_dict.keys():
                value_dict[sub_key] = value_dict[sub_key] / total_sum
            self.sorted_probs[key] = sorted(value_dict.items(),key=lambda x: x[1])

    def generate_sent(self):
        """Randomly generate a sentence."""

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """