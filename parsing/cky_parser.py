from collections import defaultdict
from itertools import product
from nltk.grammar import Nonterminal
from nltk.tree import Tree


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """

        self.grammar = grammar

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        n = len(sent)
        self._pi = defaultdict(lambda: defaultdict(float))
        self._bp = defaultdict(lambda: defaultdict(Tree))

        for i, w in enumerate(sent, start=1):
            for prod in self.grammar.productions(rhs=w):
                self._pi[(i, i)][prod.lhs().symbol()] = prod.logprob()
                self._bp[(i, i)][prod.lhs().symbol()] = Tree(prod.lhs().symbol(), [w])

        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                for s in range(i, j):
                    lbranch = self._pi[(i, s)]
                    rbranch = self._pi[(s + 1, j)]
                    combinations = product(lbranch, rbranch)
                    for c in combinations:
                        prods = [prod for prod in self.grammar.productions(rhs=Nonterminal(c[0])) if tuple(map(lambda x: x.symbol(), prod.rhs())) == c]
                        for prod in prods:
                            last_prob = self._pi[(i, j)].get(prod.lhs().symbol(), float('-inf'))
                            current_prob = prod.logprob() + lbranch[c[0]] + rbranch[c[1]]
                            if current_prob > last_prob:
                                self._pi[(i, j)][prod.lhs().symbol()] = current_prob
                                left_bps = self._bp[(i, s)][c[0]]
                                right_bps = self._bp[(s + 1, j)][c[1]]
                                self._bp[(i, j)][prod.lhs().symbol()] = Tree(prod.lhs().symbol(), [left_bps, right_bps])

        return (self._pi[(1, n)]['S'], self._bp[(1, n)]['S'])
