"""Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # tag
    hits, total = 0, 0
    unk_hits, unk_total = 0, 0
    known_hits, known_total = 0, 0
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        model_words_tags = zip(word_sent, model_tag_sent)

        unk_words_gold_tags = [(word, tag) for (word, tag)
                               in sent if model.unknown(word)]
        unk_words_tags = [(word, tag) for (word, tag)
                          in model_words_tags if model.unknown(word)]

        unk_hits_sent = [m == g for m, g
                         in zip(unk_words_tags, unk_words_gold_tags)]
        unk_hits += sum(unk_hits_sent)
        unk_total += len(unk_words_tags)
        unk_acc = float(unk_hits) / unk_total

        # global score
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits += sum(hits_sent)
        total += len(sent)
        acc = float(hits) / total

        known_hits += sum(hits_sent) - sum(unk_hits_sent)
        known_total += (len(sent) - len(unk_words_tags))
        known_acc = float(known_hits) / known_total

        progress('{:3.1f}%({:2.2f}%,{:2.2f}%,{:2.2f}%)'
                 .format(float(i) * 100 / n,
                         acc * 100, known_acc * 100, unk_acc * 100))

    acc = float(hits) / total

    print('')
    print('Accuracy: {:2.2f}%'.format(acc * 100))
