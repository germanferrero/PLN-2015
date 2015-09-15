"""Seperate sents for tests.

Usage:
  separate.py -p <p> -o <file>
  separate.py -h | --help

Options:
  -p <p>        Percentage of sents destined to testing.
  -o <file>     Output sentences file.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle

from nltk.corpus import gutenberg


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    sents = gutenberg.sents('austen-emma.txt')

    # train the model
    p = int(opts['-p'])

    total_sents = len(sents)
    n_test_sents = int(total_sents * (1/p))
    test_sents = list(sents[-n_test_sents:])
    # save it
    filename = opts['-o']
    with open(filename, 'wb') as f:
        pickle.dump(test_sents, f)
