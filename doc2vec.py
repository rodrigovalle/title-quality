import numpy as np

from sklearn.model_selection import StratifiedKFold
from gensim.models import doc2vec
from gensim.sklearn_api.d2vmodel import D2VTransformer
from gensim.utils import simple_preprocess
from helper import load_data

import pdb

def read_titles(X):
    for i, title in enumerate(X.title):
        yield doc2vec.TaggedDocument(simple_preprocess(title), [i])

def read_titles_tagged(X):
    for title, c1, c2, c3 in zip(X.title, X.category1, X.category2, X.category3):
        yield doc2vec.TaggedDocument(simple_preprocess(title), [c1, c2, c3])

def train(X):
    d2v = doc2vec.Doc2Vec(vector_size=300, window=10, min_count=1, workers=4, epochs=55)
    d2v.build_vocab(read_titles(X))
    d2v.train(read_titles_tagged(X), total_examples=d2v.corpus_count, epochs=d2v.epochs)
    return d2v

if __name__ == '__main__':
    dataset = load_data()
    data = dataset['data']
    concise_lbl = dataset['concise_label']
    clarity_lbl = dataset['clarity_label']
    both_lbl = np.stack([concise_lbl, clarity_lbl], axis=1)
    strat_lbl = np.apply_along_axis(lambda row: sum(e*2**i for i, e in enumerate(row)), 1, both_lbl)

    skf = StratifiedKFold(n_splits=5)
    train_idx, test_idx = next(skf.split(data.title, strat_lbl))

    d2v = train(data.iloc[train_idx])
