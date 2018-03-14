import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold

ext = '.pickle'
features = ['country', 'sku_id', 'title', 'category1', 'category2', 'category3',
            'description', 'price', 'product_type']

def load_data():
    # training, test, and validation data
    data = pd.read_csv('data_train.csv', header=None, names=features)

    # labels we're trying to predict
    concise_label = pd.read_csv('conciseness_train.labels', header=None).values.ravel()
    clarity_label = pd.read_csv('clarity_train.labels', header=None).values.ravel()

    # use StratifiedKFold train/test split to conserve label distribution
    kf = StratifiedKFold(n_splits=5, random_state=1)
    concise_train, concise_test = next(kf.split(data, concise_label))
    clarity_train, clarity_test = next(kf.split(data, clarity_label))

    concise_train_X = data.iloc[concise_train]
    concise_train_y = concise_label[concise_train]
    concise_test_X = data.iloc[concise_test]
    concise_test_y = concise_label[concise_test]

    clarity_train_X = data.iloc[clarity_train]
    clarity_train_y = clarity_label[clarity_train]
    clarity_test_X = data.iloc[clarity_test]
    clarity_test_y = clarity_label[clarity_test]

    # we have these data files but they're useless without labels
    #validation = pd.read_csv('data_valid.csv', header=None, names=features)
    #test = pd.read_csv('data_test.csv', header=None, names=features)

    return {
        'concise_train': (concise_train_X, concise_train_y),
        'concise_test': (concise_test_X, concise_test_y),
        'clarity_train': (clarity_train_X, clarity_train_y),
        'clarity_test': (clarity_test_X, clarity_test_y)
    }


def load_obj(objname):
    try:
        with open(objname + ext, 'rb') as dumpfile:
            return pickle.load(dumpfile)
    except (IOError, FileNotFoundError):
        return None


def caching_trainer(train_fn, filename, retrain=False):
    """Cache expensive training by saving to a file.

    parameters
    train_fn: function that performs the expensive training.
    filename: file to check for an existing model or save a new model.
    retrain: set to true to force training.
    """
    def train_and_cache():
        estimator = train_fn()
        with open(filename + ext, 'wb') as dumpfile:
            print('saving model to ' + filename + ext, flush=True)
            pickle.dump(estimator, dumpfile, pickle.HIGHEST_PROTOCOL)
        return estimator

    if retrain:
        return train_and_cache()

    try:
        with open(filename + ext, 'rb') as dumpfile:
            estimator = pickle.load(dumpfile)
            print('loaded model from ' + filename + ext, flush=True)
            return estimator
    except (IOError, FileNotFoundError):
        return train_and_cache()
