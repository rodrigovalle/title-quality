import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction

features = ['country', 'sku_id', 'title', 'category1', 'category2', 'category3',
            'description', 'price', 'product_type']

# training, test, and validation data
train = pd.read_csv('data_train.csv', header=None, names=features)
test = pd.read_csv('data_test.csv', header=None, names=features)
validation = pd.read_csv('data_valid.csv', header=None, names=features)

# labels we're trying to predict
concise_train = pd.read_csv('conciseness_train.labels', header=None)
clarity_train = pd.read_csv('clarity_train.labels', header=None)

# TODO: convert each row to a numerical vector
# (use sklearn.feature_extraction)

# train an SVM binary classifier for each of these labels
# hyperparameter documentation found at:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
concise_cls = svm.SVC(C=1.0, kernel='rbf', gamma='auto', tol=1e-3,
                      probability=True)
clarity_cls = svm.SVC(C=1.0, kernel='rbf', gamma='auto', tol=1e-3,
                      probability=True)
