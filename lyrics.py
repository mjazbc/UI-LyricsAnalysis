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
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib


def draw_length_histogram(lyrics_lengths, filename):
    plt.clf()
    print(sorted(set(lyrics_lengths)))
    print(np.mean(lyrics_lengths))
    print(max(lyrics_lengths))
    bins = np.arange(0, 6500, 100)
    lyrics_lengths = np.nan_to_num(lyrics_lengths)
    plt.hist(lyrics_lengths, bins, histtype='bar', rwidth=0.8)
    plt.axvline(100, color='r', linestyle='dashed', linewidth=1)
    plt.xlabel("lengths")
    plt.tight_layout()
    ylims = plt.ylim()
    plt.ylim((0.1, ylims[1]))

    plt.savefig(filename)


def prepare_data(path, columns, save_to_path):
    # check if we need sentinels!
    _df = pd.read_csv(path, names=columns, skiprows=1)

    print("whole")
    print(len(_df))
    lyrics_lenghts = _df.lyrics.str.len()
    empty_genre = _df.genre.notnull()  # if all five tags are empty, genre is NaN
    print("empty genre")
    print(len(_df) - len(_df.loc[empty_genre]))
    print("<100 chars")
    print(len(_df.loc[lyrics_lenghts < 100.]))
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

    draw_length_histogram(_df.lyrics.str.len(), "original_lengths.png")
    draw_length_histogram(df.lyrics.str.len(), "filtered_lengths.png")

    # just checking if data makes sense
    print(pd.unique(df.genre.values))
    print(pd.unique(df.era.values))
    print(pd.unique(df.year.values))

    # df.to_csv(save_to_path, sep=";")


def word_tuples(s, k):
    """Create and return generator from s, length k"""
    for i in range(len(s) - k + 1):
        yield ' '.join(s[i:i + k])

def get_matrix(data, k):
    t1 = time.time()
    vectorizer = CountVectorizer(ngram_range=(1,k))
    vectorizer.fit(data.lyrics)
    vector = vectorizer.transform(data.lyrics)
    print('Vectorization:\t',time.time() -t1)
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

    data = get_matrix(df, 3)
    data = data.asfptype()

    n_samples, n_features = data.shape
    n_digits = 6
    labels = list(df.genre)
    unique_labels = set(labels)

    #reduce dimensions to enable plotting
    svd = decomposition.TruncatedSVD(n_components=2, algorithm='arpack')
    svd.fit(data)
    reduced_data = svd.transform(data)
    print('Expl. variance:\t',sum(svd.explained_variance_ratio_)*100, '%')
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    palette = matplotlib.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_dict = {color:label for color, label in zip(unique_labels, palette)}
    colors = [color_dict[label] for label in labels]

    plt.scatter(reduced_data[:,0],reduced_data[:,1], c=colors)
    # plt.legend(ncol = 3,  loc='lower right',)
    plt.show()

if __name__ == "__main__":
    t1 = time.time()

    # data preprocess
    clmns = ["rank", "song", "artist", "year", "lyrics", "source",
             "tag1", "tag2", "tag3", "tag4", "tag5", "genre", "era"]
    prepare_data(path="data/billboard_lyrics_1964-2017_tagged.csv", columns=clmns, save_to_path="data/preprocessed.csv")
    sys.exit(1)

    df = pd.read_csv("data/preprocessed.csv", sep=';')
    df = df[:100]

    # get_matrix(df, 1)
    # clustering
    run_kmeans(df, k=3)

    print(time.time() - t1)
