from nltk.grammar import induce_pcfg
from nltk.grammar import Nonterminal
from parsing.cky_parser import CKYParser
from parsing.util import unlexicalize, lexicalize
from parsing.baselines import Flat


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        self.start = start

        unlex_sents = []
        for psent in parsed_sents:
            ul_sent = unlexicalize(psent.copy(deep=True))
            ul_sent.collapse_unary(collapsePOS=True, collapseRoot=True)
            unlex_sents.append(ul_sent)

        all_prods = []
        for ul_sent in unlex_sents:
            all_prods += ul_sent.productions()

        grammar = induce_pcfg(start=Nonterminal(start), productions=all_prods)
        self.prods = grammar.productions()
        self.parser = CKYParser(grammar)

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self.prods

    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        tree = self.parser.parse(tags)[1]
        if tree:
            tree.un_chomsky_normal_form()
            tree = lexicalize(tree, sent)
        else:
            flat = Flat(parsed_sents=None, start=self.start)
            tree = flat.parse(tagged_sent)
        return tree
