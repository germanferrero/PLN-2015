"""Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  interpolation: N-grams with interpolation smoothing.
                  backoff: N-grams with backoff smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import gutenberg

from languagemodeling.ngram import *


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    sents = gutenberg.sents('austen-emma.txt')

    total_sents = len(sents)
    n_training_sents = int(total_sents * (0.9))
    sents = list(sents[:n_training_sents])

    # train the model
    n = int(opts['-n'])

    m = opts['-m']
    if m == 'addone':
        model = AddOneNGram(n, sents)
    elif m == 'backoff':
        model = BackOffNGram(n, sents)
    elif m == 'interpolation':
        model = InterpolatedNGram(n, sents)
    else:
        model = NGram(n, sents)
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
