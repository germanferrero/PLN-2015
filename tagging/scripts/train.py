"""Train a sequence tagger.

Usage:
  train.py [-m <model>] -o <file> [-n <n>]
  train.py -h | --help

Options:
  -m <model>    Model to use [default: hmm]:
                  base: Baseline
                  hmm: MLHMM
  -n <n>        Grams size
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM

models = {
    'base': BaselineTagger,
    'hmm': MLHMM
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    n = opts['-n']
    # train the model
    if n is not None:
        model = models[opts['-m']](int(n), sents)
    else:
        model = models[opts['-m']](sents)
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
