"""Train a sequence tagger.

Usage:
  train.py [-m <model>] -o <file> [-n <n>] [-c <classifier>]
  train.py -h | --help

Options:
  -m <model>    Model to use [default: hmm]:
                  base: Baseline
                  hmm: MLHMM
                  memm: MEMM
  -n <n>        Grams size
  -o <file>     Output model file.
  -h --help     Show this screen.
  -c <classifier> classifier for MEMM:
                    logreg : LogisticRegression
                    mnb: MultinomialNB
                    svn: LinearSVC
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM
from tagging.memm import MEMM

models = {
    'base': BaselineTagger,
    'hmm': MLHMM,
    'memm': MEMM
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    n = opts['-n']
    c = opts['-c']
    # train the model
    if n is not None:
        if c is not None:
            model = models[opts['-m']](int(n), sents, c)
        else:
            model = models[opts['-m']](int(n), sents)
    else:
        model = models[opts['-m']](sents)
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
