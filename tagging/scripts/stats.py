"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

from corpus.ancora import SimpleAncoraCorpusReader

from collections import Counter
import itertools

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    sents = list(corpus.tagged_sents())
    words_tags = list(itertools.chain(*sents))
    freq_words_tags = Counter(words_tags)
    words_amount = len(words_tags)
    words, tags = zip(*words_tags)
    words_voc = set(words)
    tags_voc = set(tags)
    freq_tags = Counter(tags)

    # compute the statistics
    print ('Basic information')
    print ('sents: {}'.format(len(sents)))
    print ('words(total): {}'.format(words_amount))
    print ('words(vocabulary): {}'.format(len(words_voc)))
    print ('tags(vocabulary): {}'.format(len(tags_voc)))
    print ('')
    print ('Tags Info:')

    frequents_tags = freq_tags.most_common(10)

    print('More frequents tags: {}'.format([count[0] for count in frequents_tags]))
    print('')
    for tag, count in frequents_tags:
        print('Tag: {}'.format(tag))
        print('ocurrences: {}'.format(count))
        print('percentaje: {}%'.format(100*(count/len(tags))))
        common_words_count = sorted([fwt for fwt in list(freq_words_tags.items())
                                     if fwt[0][1] == tag],
                                    key=lambda x: x[1],
                                    reverse=True)[:5]
        print('Common words: {}'.format([cwc[0][0] for cwc in common_words_count]))
