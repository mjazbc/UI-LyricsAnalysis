import logging
import pickle
import re
import time

import nltk
import numpy as np
import pandas as pd
import wordninja
import gensim
from gensim.models import Word2Vec
from nltk.corpus import words, brown
from nltk.sentiment.vader import SentimentIntensityAnalyzer, negated
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix, hstack
from sklearn import metrics, preprocessing
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import normalize, MultiLabelBinarizer, label_binarize
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import xgboost as xgb


# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
# nltk.download('words')

# logging.basicConfig(level=logging.INFO)


def save_obj(obj, name):
    print("saving to %s" % name)
    pickle.dump(obj, open(name, 'wb'))


def load_obj(path):
    print("loading from %s" % path)
    return pickle.load(open(path, 'rb'))


def plot_cm(cm, labels, classifier, normalize=True, filename=""):

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
    plt.savefig(filename)


class LyricsAnalysis:
    def __init__(self, fp):
        """
        Loads lyrics dataset (.csv) file and parses the dataset.
        :param fp: filepath of dataset
        :return:
            corpus: (lyrics id: lyrics text)
        """
        class_name = "genre"
        self.w2v = {}
        self.tokenized = {}

        df = pd.read_csv(fp, delimiter=";")
        df = df[df.genre != 'other']
        # df = df.head(100)
        print(df[class_name].value_counts())
        self.corpus = pd.Series(df.lyrics.values).to_dict()
        le = preprocessing.LabelEncoder().fit(df[class_name])
        self.y = le.transform(df[class_name])
        self.text = list(df.lyrics.values)
        
        self.splitIdx = len(df.index[df.year < 2010])

        now = time.time()
        self.ie_preprocess()
        print('Preprocessing: ' + '%0.2f' % (time.time() - now))

        self.sid = SentimentIntensityAnalyzer()

        self.songs = df.song.values
        self.genre = df.genre.values
        self.artist = df.artist.values

    def ie_preprocess(self):
        tokenizer = TweetTokenizer(reduce_len=True)

        for id, text in self.corpus.items():
            tokenized = tokenizer.tokenize(text)
            self.tokenized[id] = tokenized

    #  ############################## FEATURIZING ##############################

    # SEMANTIC

    def word2vecVectorizer(self):
        sentencesData=[] #vocabulary
        for g in self.tokenized.values():
            sentencesData.append(set(g))
        
        sentencesData.append(words.words())

        model = gensim.models.Word2Vec(sentencesData, size=50,window=5,min_count=5)
        vocab=model.wv.vocab
        vocab=(set(vocab))
        word2vecTokens=[] #word2vec for all the data

        for g in self.tokenized.values(): #running loop over all the tokens
            vc=[] #to store temp vector for each token in an instance
            for s in g:
                if (s in vocab):
                    vc.append(model.wv[s])
            word2vecTokens.append(vc) # appending all the vectors of an instance
        
        g=[]
        for i in range (len(word2vecTokens)):
            g.append(np.sum(word2vecTokens[i], axis=0)/len(word2vecTokens[i]))

        return np.matrix(g)
        # columns=[]
        # index=[]
        # for k in range(0,80000):
        #     index.append(k)
        # for i in range(1,51):
        #     columns.append("w2v_"+str(i))
        # dar=pd.DataFrame(g,columns=columns)
        # dar.head()

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
            # text = self.connect_negated(text)
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

    def kmers(self, s, k=3):
        for i in range(len(s) - k + 1):
            yield tuple(s[i:i + k])

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

    def word_counter(self, tokenizer):
        cntr = csr_matrix((len(self.text), 3))
        for id, lrcs in enumerate(self.text):
            length = len(tokenizer(lrcs))
            uniq = len(set(tokenizer(lrcs)))
            cntr[id, 0] = length
            cntr[id, 1] = uniq
            cntr[id, 2] = uniq / length #ratio
        print(cntr[id,2])

        return normalize(cntr, axis=0)

    def get_years(self):
        years = label_binarize(self.years, list(set(self.years)))
        return years

    # FEATURIZING

    def featurize(self, bow):
        '''
        Tokenizes and creates TF-IDF BoW vectors.
        :param corpus: A list of strings each string representing document.
        :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
        '''
        start = time.time()
        now = start

        tokenizer = TweetTokenizer(reduce_len=True).tokenize
        wordnet = nltk.WordNetLemmatizer()
        c = 0
        c_changed = 0
        for index, text in enumerate(self.text):
            build_tmp_text = []
            for word in tokenizer(text):
                lem1 = wordnet.lemmatize(word)
                if word == lem1:
                    lem2 = wordnet.lemmatize(word, pos='v')
                    if word == lem2:
                        build_tmp_text.append(word)
                    else:
                        build_tmp_text.append(lem2)
                        c_changed += 1
                else:
                    build_tmp_text.append(lem1)
                    c_changed += 1
                c += 1
            self.text[index] = " ".join(build_tmp_text)
        print(float(c_changed) / c)
        print('Lemmatization: ' + '%0.2f' % (time.time() - now))
        now = time.time()
        # TODO: consider max features!!!
        vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer,
                                     stop_words="english", ngram_range=bow, min_df=2)
        X = csr_matrix(vectorizer.fit_transform(self.text))
        feature_names = vectorizer.get_feature_names()
        print('Tfidf: ' + '%0.2f' % (time.time() - now))

        print(X.shape)
        # to manually check if the tokens are reasonable
        # print(vectorizer.get_feature_names())

        # now = time.time()
        # floodingCharDict = self.character_flooding()
        # floodingPuncDict = self.punctuation_flooding()
        # print('Flooding: ' + '%0.2f' % (time.time() - now))

        # X = hstack([X, floodingCharDict])
        # X = hstack([X, floodingPuncDict])

        now = time.time()
        sent = self.sentiment_features()
        print('Sentimality: ' + '%0.2f' % (time.time() - now))
        X = hstack([X, sent])
        feature_names += ['@overall_score', '@difference', '@polarity']

        # seman = self.semantic_features()
        # print('Semantic: ' + '%0.2f' % (time.time() - now))
        # X = hstack([X, seman])

        # add number of words (scaled 0-1)
        lyrcs_lengths = self.word_counter(tokenizer=tokenizer)
        X = hstack([X, lyrcs_lengths])
        feature_names += ['@length', '@unique_words', '@unique_ratio']
        print('word counting: ' + '%0.2f' % (time.time() - now))

        # add year binarized
        # years = self.get_years()
        # X = hstack([X, years])

        w2v = self.word2vecVectorizer()

        X = hstack([X, w2v])
        feature_names += ['w2v_'+str(i) for i in range(50)]
        print('Total featurize: ' + '%0.2f' % (time.time() - start))
        return X, feature_names


def get_predict_topn(y_true, y_pred, topn=None, topnprob=0.1):
    """If topn set (integer), it will return predict array based on absolute topN elements.
       If topnprob set (float), return predict array based on relative (10%) topN elements."""
    new_predict = []
    for row, pred in enumerate(y_pred):
        if topn:
            top2 = pred.argsort()[-topn:][::-1]
            if y_true[row] in top2:
                new_predict.append(y_true[row])
            else:
                new_predict.append(top2[0])
        else:
            top1 = pred.argsort()[-1:][0]
            topnprobs = np.where(y_pred[row] > y_pred[row, top1] - topnprob)[0]
            if y_true[row] in topnprobs:
                new_predict.append(y_true[row])
            else:
                new_predict.append(top1)
    return np.array(new_predict)


def get_errors(y_true, y_pred):
    errs = []
    for row, pred in enumerate(y_pred):
        pred_class = pred.argsort()[-1:][0]
        if pred_class != y_true[row]:
            errs.append(y_pred[row, pred_class] - y_pred[row, y_true[row]])
        else:
            errs.append(0)
    return errs


def print_top10_features(feature_names, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print('%s: %s' % (class_label, ', '.join(feature_names[j] for j in top10)))


if __name__ == "__main__":
    # Experiment settings
    print('--- Started ----')
    DATASET_FP = "data/preprocessed_fixed.csv"

    K_FOLDS = 10  # crossvalidation

    TOPNERR = 0.0

    for bow in [(1, 1), (2, 2), (1, 2), (3, 3), (1, 3)]:
        print(bow)
        # Loading dataset and featurised simple Tfidf-BoW model
        idt = LyricsAnalysis(DATASET_FP)
        # corpus, y = idt.parse_dataset_old(DATASET_FP)

        # vecs = idt.word2vecVectorizer()

        X, feature_names = idt.featurize(bow=bow)

        train = X.tocsr()[0:idt.splitIdx, :]
        train_y = idt.y[0:idt.splitIdx]

        test = X.tocsr()[idt.splitIdx:,:]
        test_y = idt.y[idt.splitIdx:]

        class_counts = np.asarray(np.unique(idt.y, return_counts=True)).T.tolist()
        # print("Num of classes: ", class_counts)

        # method = "predict"
        method = "predict_proba"
        if method == "predict_proba":
            algs = [svm.SVC(kernel='linear', probability=True), RandomForestClassifier(n_estimators=1000, n_jobs=-1), svm.SVC(kernel='rbf', probability=True)]
            algs_names = ["svc_linear", "rf", "svc_rbf"]
        else:
            algs = [svm.SVC(kernel='linear', probability=False)]
            algs_names = ["svc_linear"]
            # algs = [LinearSVC(multi_class="crammer_singer")]

        randpred = np.random.randint(len(np.unique(idt.y)), size=len(idt.y))
        acc = metrics.accuracy_score(idt.y, randpred)
        prec = metrics.precision_recall_fscore_support(idt.y, randpred)

        print("Random classifier:")
        print('Accuracy', acc, '| Precision', np.mean(
            prec[0]), '| Recall', np.mean(prec[1]), '| F-score', np.mean(prec[2]))

        DUMMYCLF = DummyClassifier(strategy="constant", constant=3)
        majority_predict = cross_val_predict(DUMMYCLF, X, idt.y, cv=10, method="predict")
        acc = metrics.accuracy_score(idt.y, majority_predict)
        prec = metrics.precision_recall_fscore_support(idt.y, majority_predict)

        print()
        print("Dummy classifier majority:")
        print('Accuracy', acc, '| Precision', np.mean(
            prec[0]), '| Recall', np.mean(prec[1]), '| F-score', np.mean(prec[2]))

        for alg_idx, CLF in enumerate(algs):
            print(CLF.__class__.__name__)
            now = time.time()

            predicted = cross_val_predict(CLF, X, idt.y, cv=K_FOLDS, method=method)
            cm_labels = ['country', 'dance', 'hiphop', 'pop', 'rock', 'soul']
            # print_top10_features(feature_names,CLF, cm_labels)

            if method == "predict_proba":
                # predicted = get_predict_topn(idt.y, predicted, topn=2)
                prob_errors = pd.Series(get_errors(idt.y, predicted))
                predicted = get_predict_topn(idt.y, predicted, topnprob=TOPNERR)
                print("average err: %.4f" % np.mean(prob_errors))
                print("10 largest errors:")
                for idx, err in prob_errors.nlargest(10).iteritems():
                    print("\t%s, %s, %s, %s, %.3f" % (idt.songs[idx], idt.artist[idx], idt.genre[idx], cm_labels[predicted[idx]], err))

            acc = metrics.accuracy_score(idt.y, predicted)
            prec = metrics.precision_recall_fscore_support(idt.y, predicted)

            print('Full')
            print('Accuracy', acc, '| Precision', np.mean(prec[0]), '| Recall', np.mean(prec[1]), '| F-score',
                  np.mean(prec[2]))

            acc = metrics.accuracy_score(idt.y, predicted)
            prec = metrics.precision_recall_fscore_support(idt.y, predicted)

            conf_matrix = confusion_matrix(idt.y, predicted)
            plot_cm(conf_matrix, cm_labels, CLF.__class__.__name__, True, "new_cm_%s_%.0f_bow%d-%d.png" % (algs_names[alg_idx], TOPNERR*100, bow[0], bow[-1]))
