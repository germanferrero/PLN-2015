# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from languagemodeling.constants import BEGIN, END
from math import log
import random
import abc


class LangModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n, sents):
        self.n = n
        my_set = set()
        for sent in sents:
            for token in sent:
                if token != BEGIN:
                    my_set.add(token)
        # We also count END token for out Vocabulary size
        self.v_size = len(my_set) + 1

    @abc.abstractmethod
    def count(self, tokens):
        """
        Count for a k-gram
        """
        return

    @abc.abstractmethod
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        return

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        prob = 1
        my_sent = ([BEGIN] * (self.n - 1)) + sent
        my_sent.append(END)
        for i in range(self.n-1, len(my_sent)):
            prob *= self.cond_prob(my_sent[i], my_sent[i-(self.n-1):i])
            if prob == 0:
                break
        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        prob = 0.0
        my_sent = ([BEGIN] * (self.n - 1)) + sent
        my_sent.append(END)
        for i in range(self.n-1, len(my_sent)):
            sent_prob = self.cond_prob(my_sent[i], my_sent[i-(self.n-1):i])
            if sent_prob == 0:
                return float('-inf')
            else:
                prob += log(sent_prob, 2)
        return prob

    def cross_entropy(self, sents):
        prob = 0.0
        M = 0
        for sent in sents:
            M += len(sent)
            sent_prob = self.sent_log_prob(sent)
            prob += sent_prob
        return - (prob / M)

    def perplexity(self, sents):
        cross_entropy = self.cross_entropy(sents)
        return 2**(cross_entropy)

    def V(self):
        """Size of the vocabulary.
        """
        return self.v_size

    def _get_count_dict_for_n(self, n, sents):
        counts = defaultdict(int)
        begin_sentence_list = [BEGIN] * (n-1)
        for sent in sents:
            # Insert sentence BEGIN and END markers.
            sent = begin_sentence_list + sent + [END]

            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
        return counts


class NGram(LangModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        super(NGram, self).__init__(n, sents)
        self.counts = self._get_count_dict_for_n(n, sents)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        assert isinstance(tokens, tuple)
        assert len(tokens) <= self.n

        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if not prev_tokens:
            prev_tokens = []
        # assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]



class NGramGenerator:

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.n = model.n
        # Initialize probs, a dict of float's dicts.
        self.probs = defaultdict(lambda: defaultdict(float))
        self.sorted_probs = defaultdict(list)
        # First count tokens appearances.
        for k, v in model.counts.items():
            if len(k) == (self.n):
                self.probs[k[:-1]][k[-1]] = v

        # Then calculate probability distribution from total appearences.
        for key, value_dict in self.probs.items():
            total_sum = sum(value_dict.values())
            for sub_key in value_dict.keys():
                value_dict[sub_key] = value_dict[sub_key] / total_sum
            self.sorted_probs[key] = sorted(value_dict.items(), key=lambda x: (-x[1], x[0]))

    def generate_sent(self):
        """Randomly generate a sentence."""
        # Init sent with begin sentence markers
        sent = [BEGIN] * (self.n - 1)
        while True:
            prev_tokens = tuple(sent[len(sent)-(self.n-1):])
            next_token = self.generate_token(prev_tokens)
            if next_token == END:
                break
            else:
                sent += [next_token]
        return sent[self.n-1:]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        probs = self.sorted_probs[prev_tokens]
        u = random.random()
        cumulative = 0
        for i in range(len(probs)):
            cumulative += probs[i][1]
            if u <= cumulative:
                return probs[i][0]


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        super(AddOneNGram, self).__init__(n, sents)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            prev_tokens = []

        tokens = prev_tokens + [token]
        return (float((self.counts[tuple(tokens)]) + 1) / (self.counts[tuple(prev_tokens)] + self.V()))



class InterpolatedNGram(LangModel):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """

        # If gamma is not provided, separate sents for development data.
        if not gamma:
            total_sents = len(sents)
            n_training_sents = int(total_sents * (0.9))
            self.held_out_sents = list(sents[-(total_sents-n_training_sents):])
            # Override sents with training sents
            sents = list(sents[:n_training_sents])

        super(InterpolatedNGram, self).__init__(n, sents)

        self.addone = addone

        # all_counts will keep a different count dicts for each k-gram model count
        # for k=1 to k=n
        self.all_counts = []
        for i in range(1,n+1):
            counts = self._get_count_dict_for_n(i,sents);
            self.all_counts.append(counts)

        # Finally estimate gamma from held out data.
        # or get it from params.
        if not gamma:
            self.estimate_gamma()
        else:
            self.gamma = gamma

    def estimate_gamma(self):
        gamma = 10
        perplexities = []
        for i in range(20):
            self.gamma = gamma
            perplexity = self.perplexity(self.held_out_sents)
            perplexities.append((gamma, perplexity))
            gamma += 100
        # Take gamma that generate best (lower) perplexity
        self.gamma = min(perplexities, key=lambda x: x[1])[0]

    def ML_cond_prob(self, token, prev_tokens=None):
        if not prev_tokens:
            prev_tokens = []
        # assert len(prev_tokens) == n - 1
        n = len(prev_tokens)

        tokens = prev_tokens + [token]
        if self.addone and n == 0:
            result = float((self.all_counts[n][tuple(tokens)] + 1)
                           / (self.all_counts[n][tuple(prev_tokens)] + self.V()))
        else:
            result = float(self.all_counts[n][tuple(tokens)]
                           / (self.all_counts[n][tuple(prev_tokens)]))
        return result

    def get_lambdas(self, gamma, tokens):
        lambdas = []
        for i in range(self.n-1):
            # Note here that we use ( self.n - i ) - 1 because
            # it give as de {self.n - i}-gram counts dict.
            count = self.all_counts[(self.n-i)-1][tuple(tokens[i:])]
            lambda_i = count / (count + gamma)
            lambda_i *= 1 - sum(lambdas)
            lambdas.append(lambda_i)
        lambdas.append(1 - sum(lambdas))
        return lambdas

    def cond_prob(self, token, prev_tokens=None):
        if not prev_tokens:
            prev_tokens = []
        lambdas = self.get_lambdas(self.gamma, prev_tokens)
        prob = 0
        for i in range(self.n):
            if lambdas[i] != 0:
                prob += lambdas[i] * self.ML_cond_prob(token, prev_tokens[i:])
        return prob

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        assert isinstance(tokens, tuple)
        assert len(tokens) <= self.n

        if len(tokens) == 0:
            i = 1
        else:
            i = len(tokens)
        count = self.all_counts[i-1][tokens]
        return count
