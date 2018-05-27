import logging
import pickle
import re
import time

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer, negated
from nltk.tokenize import WhitespaceTokenizer
from scipy.sparse import csr_matrix, hstack
from sklearn import metrics, preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO)


def save_obj(obj, name):
    print("saving to %s" % name)
    pickle.dump(obj, open(name, 'wb'))


def load_obj(path):
    print("loading from %s" % path)
    return pickle.load(open(path, 'rb'))


def plot_cm(cm, labels, classifier, normalize=True):

    print(cm)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize rows

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of ' +classifier+' classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + cm_labels)
    ax.set_yticklabels([''] + cm_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


class LyricsAnalysis:
    def __init__(self, fp):
        """
        Loads lyrics dataset (.csv) file and parses the dataset.
        :param fp: filepath of dataset
        :return:
            corpus: (lyrics id: lyrics text)
        """
        self.w2v = {}
        self.tokenized = {}

        df = pd.read_csv(fp, delimiter=";")
        print(len(df))
        df = df[df.genre != 'other']
        print(len(df))
        print(df['genre'].value_counts())
        self.corpus = pd.Series(df.lyrics.values).to_dict()
        self.labels = list(set(df.genre))
        le = preprocessing.LabelEncoder().fit(df.genre)
        self.y = le.transform(df.genre)
        self.text = list(df.lyrics.values)

        now = time.time()
        self.ie_preprocess()
        print('Preprocessing: ' + '%0.2f' % (time.time() - now))

        self.sid = SentimentIntensityAnalyzer()

    def ie_preprocess(self):
        tokenizer = WhitespaceTokenizer()

        for id, text in self.corpus.items():
            tokenized = tokenizer.tokenize(text)
            self.tokenized[id] = tokenized

    #  ############################## FEATURIZING ##############################

    # SEMANTIC

    def word2vecVectorizer(self, text, n):
        return np.array([np.mean([self.w2v[w] for w in text if w in self.w2v]
                                 or [np.zeros(n)], axis=0)])

    def semantic_features(self, n=200):
        res = csr_matrix((len(self.tokenized), n))

        sents = self.corpus.values()
        model = Word2Vec(sents, size=n)
        self.w2v = dict(zip(model.wv.index2word, model.wv.syn0))

        for id, text in self.tokenized.items():
            res[id] = self.word2vecVectorizer(text, n)

        return res

    # SENTIMENT

    def connect_negated(self, tokenized_text):
        if not negated(tokenized_text):
            return tokenized_text
        else:
            res = list()
            negated_part = ""
            in_negated = False
            for word in tokenized_text:
                if negated([word]):
                    in_negated = True
                    negated_part = word
                elif in_negated and negated([word]):
                    res.extend(negated_part.split(" "))
                    in_negated = False
                elif in_negated and self.sid.polarity_scores(word)['compound'] != 0:
                    negated_part += ' ' + word
                    in_negated = False
                    res.append(negated_part)
                elif in_negated:
                    negated_part += ' ' + word
                else:
                    res.append(word)
            return res

    def sentiment_features(self):
        res = list()
        for id, text in self.tokenized.items():
            text = self.connect_negated(text)
            scores = [self.sid.polarity_scores(
                word)['compound'] for word in text]
            if not scores:
                res.append([0, 0, 0])
                continue
            # print(scores)
            max_score = max(scores)
            min_score = min(scores)

            overall_score = self.sid.polarity_scores(self.text[id])['compound']
            difference = max_score - min_score
            polarity = int((np.sign(max_score) - np.sign(min_score)) == 2)

            res.append([overall_score, difference, polarity])

        return csr_matrix(res)

    # LEXICAL

    def character_flooding(self):
        # res = [[0] for x in range(len(self.tokenized))]
        res = csr_matrix((len(self.tokenized), 1))
        regex = re.compile(r'(\w)\1\1')

        for id, text in self.corpus.items():
            is_flooded = 0
            match = regex.search(text)
            if match:
                is_flooded = 1
            res[id] = is_flooded

        return res

    def punctuation_flooding(self):
        # res = [[0] for x in range(len(self.tokenized))]
        res = csr_matrix((len(self.tokenized), 1))
        regex = re.compile(
            r'([.\?#@+,<>%~`!$^&\(\):;]|[.\?#@+,<>%~`!$^&\(\):;]\s)\1+')

        for id, text in self.corpus.items():
            is_flooded = 0
            match = regex.search(text)
            if match:
                is_flooded = 1
            res[id] = is_flooded

        return res

    def kmers(self, s, k=3):
        for i in range(len(s) - k + 1):
            yield tuple(s[i:i + k])

    def kmers_featurize(self, k=2, elem="word"):
        i = 0
        kmer_dict = {}

        print("iterate through tokenized items")
        # build dict
        for id, text in self.tokenized.items():
            if elem == "char":
                text = self.text[id]

            for kmer in list(self.kmers(text, k=k)):
                if kmer not in kmer_dict:
                    kmer_dict[kmer] = i
                    i += 1
                    print(kmer_dict)

        # featurize
        res = csr_matrix((len(self.tokenized), len(kmer_dict)))

        for id, text in self.tokenized.items():
            if elem == "char":
                text = self.text[id]

            for kmer in list(self.kmers(text, k=k)):
                # row is id of a text, column is value in kmer_dict for this kmer
                res[id, kmer_dict[kmer]] = 1

        return res

    def word_counter(self, tokenizer):
        cntr = csr_matrix((2, len(self.text)))
        for id, lrcs in enumerate(self.text):
            cntr[0, id] = len(tokenizer(lrcs))
            cntr[1, id] = len(set(tokenizer(lrcs)))
        return normalize(cntr, axis=0)

    # FEATURIZING

    def featurize(self):
        '''
        Tokenizes and creates TF-IDF BoW vectors.
        :param corpus: A list of strings each string representing document.
        :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
        '''

        tokenizer = WhitespaceTokenizer().tokenize
        vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer,
                                     stop_words="english")
        X = csr_matrix(vectorizer.fit_transform(self.text))
        # to manually check if the tokens are reasonable
        print(vectorizer.get_feature_names())

        now = time.time()
        print('Featurize: ' + '%0.2f' % (time.time() - now))

        now = time.time()
        floodingCharDict = self.character_flooding()
        floodingPuncDict = self.punctuation_flooding()
        print('Flooding: ' + '%0.2f' % (time.time() - now))

        X = hstack([X, floodingCharDict])
        X = hstack([X, floodingPuncDict])

        now = time.time()
        # sent = self.sentiment_features()
        # print('Sentimality: ' + '%0.2f' % (time.time() - now))
        # X = hstack([X, sent])

        # seman = self.semantic_features()
        # print('Semantic: ' + '%0.2f' % (time.time() - now))
        # X = hstack([X, seman])

        # add number of words (scaled 0-1)
        # lyrcs_lengths = self.word_counter(tokenizer=tokenizer)
        # X = hstack([X, lyrcs_lengths])
        # print('word counting: ' + '%0.2f' % (time.time() - now))

        return X


if __name__ == "__main__":
    # Experiment settings
    print('--- Started ----')
    DATASET_FP = "data/preprocessed.csv"

    K_FOLDS = 3  # 10-fold crossvalidation

    # Loading dataset and featurised simple Tfidf-BoW model
    idt = LyricsAnalysis(DATASET_FP)
    # corpus, y = idt.parse_dataset_old(DATASET_FP)

    X = idt.featurize()

    class_counts = np.asarray(np.unique(idt.y, return_counts=True)).T.tolist()
    print("Num of classes: ", class_counts)

    algs = [LinearSVC(multi_class="crammer_singer"), RandomForestClassifier(
        n_estimators=100, n_jobs=-1), svm.SVC()]

    for CLF in algs:
        print(CLF.__class__.__name__)
        # CLF = LinearSVC(multi_class="crammer_singer")  # the default, non-parameter optimized linear-kernel SVM

        # SVM with created features - currently uses only tokenized words.
        predicted = cross_val_predict(CLF, X, idt.y, cv=K_FOLDS)
        randpred = np.random.randint(
            len(np.unique(idt.y)), size=len(predicted))

        cm_labels = ['country', 'dance', 'hiphop', 'pop', 'rock', 'soul']

        conf_matrix = confusion_matrix(idt.y, predicted)
        plot_cm(conf_matrix,cm_labels, CLF.__class__.__name__, True)

        acc = metrics.accuracy_score(idt.y, randpred)
        prec = metrics.precision_recall_fscore_support(idt.y, randpred)

        print("Random classifier:")
        print('Accuracy', acc, '| Precision', np.mean(
            prec[0]), '| Recall', np.mean(prec[1]), '| F-score', np.mean(prec[2]))
        # Modify F1-score calculation depending on the task
        acc = metrics.accuracy_score(idt.y, predicted)
        prec = metrics.precision_recall_fscore_support(idt.y, predicted)

        acc = metrics.accuracy_score(idt.y, predicted)
        prec = metrics.precision_recall_fscore_support(idt.y, predicted)

        print('Full')
        print('Accuracy', acc, '| Precision', np.mean(prec[0]), '| Recall', np.mean(prec[1]), '| F-score',
              np.mean(prec[2]))
        # Modify F1-score calculation depending on the task
        acc = metrics.accuracy_score(idt.y, predicted)
        prec = metrics.precision_recall_fscore_support(idt.y, predicted)
