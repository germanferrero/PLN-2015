from collections import defaultdict
from nltk.tree import Tree


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """

        self.grammar = grammar
        self.start = grammar.start()
        self.prods = defaultdict(list)
        for prod in self.grammar.productions():
            self.prods[self.symbols(prod.rhs())].append(prod)

    def symbols(self, hs):
        return tuple(str(e) for e in hs)

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        n = len(sent)
        self._pi = defaultdict(lambda: defaultdict(float))
        self._bp = defaultdict(lambda: defaultdict(Tree))

        for i, w in enumerate(sent, start=1):
            for prod in self.prods[(w,)]:
                self._pi[(i, i)][prod.lhs().symbol()] = prod.logprob()
                self._bp[(i, i)][prod.lhs().symbol()] = Tree(prod.lhs().symbol(), [w])

        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                for s in range(i, j):
                    lbranch = self._pi[(i, s)]
                    rbranch = self._pi[(s + 1, j)]
                    for lk in lbranch.keys():
                        for rk in rbranch.keys():
                            for prod in self.prods[(lk, rk)]:
                                last_prob = self._pi[(i, j)].get(prod.lhs().symbol(), float('-inf'))
                                current_prob = prod.logprob() + lbranch[lk] + rbranch[rk]
                                if current_prob > last_prob:
                                    self._pi[(i, j)][prod.lhs().symbol()] = current_prob
                                    left_bps = self._bp[(i, s)][lk]
                                    right_bps = self._bp[(s + 1, j)][rk]
                                    self._bp[(i, j)][prod.lhs().symbol()] = Tree(prod.lhs().symbol(), [left_bps, right_bps])

        start = self.start.symbol()
        if start in self._pi[(1, n)]:
            return (self._pi[(1, n)][start], self._bp[(1, n)][start])
        else:
            return None, None
