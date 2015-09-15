"""
Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle

from nltk.corpus import gutenberg

if __name__ == '__main__':
    opts = docopt(__doc__)

    filename = opts['-i']
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    sents = gutenberg.sents('austen-emma.txt')
    total_sents = len(sents)

    n_test_sents = int(total_sents * (1/10))
    test_sents = list(sents[-n_test_sents:])

    cross_entropy = model.cross_entropy(test_sents)
    perplexity = model.perplexity(test_sents)
    print ("Model " + str(model.n) + "-gram :")
    print ("Cross Entropy: " + str(cross_entropy))
    print ("Perplexity: " + str(perplexity))
