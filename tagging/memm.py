from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from featureforge.vectorizer import Vectorizer
from tagging.features import (History, word_lower, word_istitle, word_isupper,
                              word_isdigit, NPrevTags, PrevWord)
import itertools

BEGIN = BEGIN_TAG = '<s>'
END = STOP_TAG = '</s>'


class MEMM:

    def __init__(self, n, tagged_sents, classifier='logreg'):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        """
        Crear sents_histories a partir de tagged_sents
        Instanciar Pipeline de scikit
        Instanciar un Clasificador y meterlo en pipeline.
        """
        classifiers = {
            'logreg': LogisticRegression(),
            'mnb': MultinomialNB(),
            'svn': LinearSVC()
        }

        self.n = n
        words_tags = list(itertools.chain(*tagged_sents))
        words, tags = zip(*words_tags)
        self.wordset = set(words)

        features = [word_lower, word_istitle, word_isupper,
                    word_isdigit]
        nprevs_features = [NPrevTags(i) for i in range(1, self.n)]
        prev_word_features = [PrevWord(f) for f in features]

        vectorizer = Vectorizer(features
                                + nprevs_features + prev_word_features)

        self.tagger = Pipeline([('vectorizer', vectorizer),
                                ('classifier', classifiers[classifier])
                                ])

        sents_histories = list(self.sents_histories(tagged_sents))
        sents_tags = list(self.sents_tags(tagged_sents))
        self.tagger = self.tagger.fit(sents_histories, sents_tags)

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        for tagged_sent in tagged_sents:
            yield from self.sent_histories(tagged_sent)

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        if len(tagged_sent) > 0:
            sent, tags = zip(*tagged_sent)
            sent = list(sent)
            for i in range(len(tags)):
                if i == 0:
                    hist_tags = (BEGIN,) * (self.n - 1)
                    history = History(sent, hist_tags, 0)
                else:
                    hist_tags = (hist_tags + (tags[i - 1],))[1:]
                    history = History(sent, hist_tags, i)
                yield history

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        for tagged_sent in tagged_sents:
            yield from self.sent_tags(tagged_sent)

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        for (word, tag) in tagged_sent:
            yield tag

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        hist_tags = (BEGIN,) * (self.n - 1)
        tags = []
        for i in range(len(sent)):
            h = History(sent, hist_tags, i)
            tag = self.tag_history(h)
            hist_tags = (hist_tags + (tag,))[1:]
            tags.append(tag)
        return tags

    def tag_history(self, h):
        """Tag a history.

        h -- the history.
        """
        return self.tagger.predict([h])[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.wordset
