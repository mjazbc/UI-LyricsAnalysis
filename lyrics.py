import random
import time
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import unidecode
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np


def prepare_data(path, columns):
    _df = pd.read_csv(path, names=columns, skiprows=1)  # check if we need sentinels!

    lyrics_lenghts = _df.lyrics.str.len()
    empty_genre = _df.genre.notnull()  # if all five tags are empty, genre is NaN
    mask = (lyrics_lenghts > 100.) & empty_genre
    drop_columns = ['rank', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'source']
    drop_duplicates_by_columns = ['song', 'artist']
    # remove row if lyrics length has < 100 characters
    # remove row if genre is empty
    # drop columns rank, tag1 - tag5 and source
    # remove rows that are duplicates by song and artist
    df = _df.loc[mask].drop(columns=drop_columns).drop_duplicates(subset=drop_duplicates_by_columns)
    filtered = 1. - (float(len(df)) / len(_df))
    print("removed %.2f%% rows" % (round(filtered, 5) * 100))
    print(len(_df))
    print(len(df))

    # hist = _df.lyrics.str.len().diff().hist()
    # plt.show()
    # hist2 = df.lyrics.str.len().diff().hist()
    # plt.show()
    return df


def tuples(s, k):
    """Create and return generator from s, length k"""
    for i in range(len(s) - k + 1):
        yield s[i:i + k]


def get_matrix(data, k):
    """Get matrix for calculating distances."""
    triples = []
    temp = []
    matrix = []

    # loop through all languages
    for line in data.lyrics:
        tmp = unidecode.unidecode(line.decode("utf-8")).lower().replace('  ', ' ')
        lyric = "".join(c for c in tmp if c.isalpha() or c == ' ')
        t = tuples(lyric, k)  # create generator
        tmp = Counter(t)  # and count all tuples
        temp.append(tmp)  # append it to temporary list for later
        triples += list(tmp.keys())  # and save keys in another list

    triples = set(triples)  # unique all keys (tuples from every language)

    for tmp in temp:  # loop through all counters
        matrix.append([tmp[te] for te in triples])  # and append this list to matrix list

    return matrix


def run_clustering(df, matrix, rnd, k, year):
    """Run hierarchical clustering using scipy."""
    rnd = min(len(df), rnd)
    random_100 = random.sample(list(range(len(df))), rnd)
    lbls = np.array([df.iloc[i].song + " " + df.iloc[i].genre for i in random_100])
    z = linkage(np.array(matrix)[random_100], 'average', 'cosine')
    dendrogram(z, labels=lbls, orientation='right')
    plt.tight_layout()  # to see all the labels
    plt.savefig('random%d-k%d-year%d.png' % (rnd, k, year), dpi=200)


if __name__ == "__main__":
    t1 = time.time()
    # settings
    k = 3
    year = -1  # use -1 for all
    rnd = 100

    # data preprocess
    df = prepare_data(path="data/billboard_lyrics_1964-2017_tagged.csv", columns=["rank", "song", "artist", "year", "lyrics", "source", "tag1", "tag2", "tag3", "tag4", "tag5", "genre", "era"])

    if year != -1:
        df = df.loc[df.year == year]

    # print this to assert data makes sense
    print(pd.unique(df.genre.values))
    print(pd.unique(df.era.values))
    print(pd.unique(df.year.values))

    matrix = get_matrix(df, k)

    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_csv("data/data-k%d-filtered.csv" % k)

    print("matrix done")

    run_clustering(df, matrix, rnd, k, year)

    print(time.time() - t1)
