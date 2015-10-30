"""Evaulate a parser.

Usage:
  eval.py -i <file> [-m <sent_max_length>] [-n <sents_count>]
  eval.py -h | --help

Options:
  -i <file>     Parsing model file.
  -h --help     Show this screen.
  -m <sent_max_length> Eval only sents with less than m+1 words.
  -n <sents_count> Eval only first n sents.
"""

from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.util import spans


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading model...')
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    n = opts['-n']
    m = opts['-m']

    print('Loading corpus...')
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    parsed_sents = list(corpus.parsed_sents())

    if m:
        parsed_sents = [sent for sent in parsed_sents if len(sent.leaves()) <= int(m)]
    if n:
        parsed_sents = parsed_sents[:int(n)]

    print('Parsing...')
    hits, u_hits, total_gold, total_model = 0, 0, 0, 0
    n = len(parsed_sents)
    format_str = '{:3.1f}% ({}/{}) (P={:2.2f}%, R={:2.2f}%, F1={:2.2f}%)'
    progress(format_str.format(0.0, 0, n, 0.0, 0.0, 0.0))
    for i, gold_parsed_sent in enumerate(parsed_sents):
        tagged_sent = gold_parsed_sent.pos()

        # parse
        model_parsed_sent = model.parse(tagged_sent)

        # compute labeled scores
        gold_spans = spans(gold_parsed_sent, unary=False)
        model_spans = spans(model_parsed_sent, unary=False)
        hits += len(gold_spans & model_spans)
        total_gold += len(gold_spans)
        total_model += len(model_spans)

        # compute unlabeled scores
        u_gold_spans = set(list(map(lambda x: x[1:], list(gold_spans))))
        u_model_spans = set(list(map(lambda x: x[1:], list(model_spans))))
        u_hits += len(u_gold_spans & u_model_spans)

        # compute labeled partial results
        prec = float(hits) / total_model * 100
        rec = float(hits) / total_gold * 100
        f1 = 2 * prec * rec / (prec + rec)

        # compute unlabeled partial results
        u_prec = float(u_hits) / total_model * 100
        u_rec = float(u_hits) / total_gold * 100
        u_f1 = 2 * u_prec * u_rec / (u_prec + u_rec)

        progress(format_str.format(float(i+1) * 100 / n, i+1, n, prec, rec, f1))

    print('')
    print('Parsed {} sentences'.format(n))
    print('Labeled')
    print('  Precision: {:2.2f}% '.format(prec))
    print('  Recall: {:2.2f}% '.format(rec))
    print('  F1: {:2.2f}% '.format(f1))
    print('')
    print('Unlabeled')
    print('  Precision: {:2.2f}% '.format(u_prec))
    print('  Recall: {:2.2f}% '.format(u_rec))
    print('  F1: {:2.2f}% '.format(u_f1))
