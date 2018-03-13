import numpy as np
import pandas as pd
import pickle

ext = '.pickle'
features = ['country', 'sku_id', 'title', 'category1', 'category2', 'category3',
            'description', 'price', 'product_type']

def load_data():
    # training, test, and validation data
    data = pd.read_csv('data_train.csv', header=None, names=features)
    split_idx = len(data) - 9000 # amount to split for test data

    data.drop(columns=['sku_id'], inplace=True)

    train = data.iloc[:split_idx, :]
    test = data.iloc[split_idx:, :]

    # we have these data files but they're useless without labels
    #validation = pd.read_csv('data_valid.csv', header=None, names=features)
    #test = pd.read_csv('data_test.csv', header=None, names=features)

    # labels we're trying to predict
    concise_label = pd.read_csv('conciseness_train.labels', header=None)
    clarity_label = pd.read_csv('clarity_train.labels', header=None)

    # split them into train and validation data sets
    concise_train = np.ravel(concise_label.iloc[:split_idx, :].values)
    clarity_train = np.ravel(clarity_label.iloc[:split_idx, :].values)
    concise_test = np.ravel(concise_label.iloc[split_idx:, :].values)
    clarity_test = np.ravel(clarity_label.iloc[split_idx:, :].values)

    return {
        'data': (train, test),
        'concise_label': (concise_train, concise_test),
        'clarity_label': (clarity_train, clarity_test)
    }


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
    except (IOError, FileExistsError):
        return train_and_cache()
