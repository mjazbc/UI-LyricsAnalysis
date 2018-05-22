import random
import sys
import time
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import unidecode
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def prepare_data(path, columns, save_to_path):
    # check if we need sentinels!
    _df = pd.read_csv(path, names=columns, skiprows=1)

    lyrics_lenghts = _df.lyrics.str.len()
    empty_genre = _df.genre.notnull()  # if all five tags are empty, genre is NaN
    mask = (lyrics_lenghts > 100.) & empty_genre
    drop_columns = ['rank', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'source']
    drop_duplicates_by_columns = ['song', 'artist']
    # remove row if lyrics length has < 100 characters
    # remove row if genre is empty
    # drop columns rank, tag1 - tag5 and source
    # remove rows that are duplicates by song and artist
    df = _df.loc[mask].drop(columns=drop_columns).drop_duplicates(
        subset=drop_duplicates_by_columns)
    # make all string lowercase, use only alphanumeric characters, skip other
    for col in ['song', 'artist', 'lyrics']:
        df[col] = df[col].apply(lambda x: "".join(
            c for c in unidecode.unidecode(x).lower() if c.isalnum() or c == ' '))
    filtered = 1. - (float(len(df)) / len(_df))
    print("removed %.2f%% rows" % (round(filtered, 5) * 100))
    print(len(_df))
    print(len(df))

    # hist = _df.lyrics.str.len().diff().hist()
    # plt.show()
    # hist2 = df.lyrics.str.len().diff().hist()
    # plt.show()

    # just checking if data makes sense
    print(pd.unique(df.genre.values))
    print(pd.unique(df.era.values))
    print(pd.unique(df.year.values))

    df.to_csv(save_to_path, sep=";")


def word_tuples(s, k):
    """Create and return generator from s, length k"""
    for i in range(len(s) - k + 1):
        yield ' '.join(s[i:i + k])



def get_matrix(data, k):
    t1 = time.time()
    """Get matrix for calculating distances."""
    triples = []
    temp = []

    # list of text documents
    # text = ["The quick brown fox jumped over the lazy dog."]
    # create the transform
    vectorizer = CountVectorizer(ngram_range=(1,k))
    # tokenize and build vocab
    vectorizer.fit(data.lyrics)
    # summarize
    print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(data.lyrics)
    # summarize encoded vector
    print(vector.shape)
    return vector

def run__hierarchical_clustering(df, k, year):
    """Run hierarchical clustering using scipy."""
    if year != -1:
        df = df.loc[df.year == year]
    matrix = get_matrix(df, 1)
    print("matrix done")
    lbls = np.array([row['song'] + " " + row['genre'] for i, row in df.iterrows()])
    z = linkage(matrix.toarray(), 'average', 'cosine')
    dendrogram(z, labels=lbls, orientation='right')
    plt.tight_layout()  # to see all the labels
    plt.savefig('k%d-year%d.png' % (k, year), dpi=200)

def run_kmeans(df, k):

    data = get_matrix(df, 1).toarray()

    n_samples, n_features = data.shape
    n_digits = 6
    labels = df.genre

    print(82 * '_')

    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
            'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == "__main__":
    t1 = time.time()

    # data preprocess
    clmns = ["rank", "song", "artist", "year", "lyrics", "source",
             "tag1", "tag2", "tag3", "tag4", "tag5", "genre", "era"]
    # prepare_data(path="data/billboard_lyrics_1964-2017_tagged.csv", columns=clmns, save_to_path="data/preprocessed.csv")
    # sys.exit(1)

    df = pd.read_csv("data/preprocessed.csv", sep=';')
    df = df[:100]

    # get_matrix(df, 1)
    # clustering
    run_kmeans(df, k=3)

    print(time.time() - t1)
