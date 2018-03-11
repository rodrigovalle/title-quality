import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error

features = ['country', 'sku_id', 'title', 'category1', 'category2', 'category3',
            'description', 'price', 'product_type']

# training, test, and validation data
data = pd.read_csv('data_train.csv', header=None, names=features)
split_idx = len(data) - 10000 # amount to split for validation data

data.drop(columns=['sku_id'], inplace=True)

train = data.iloc[:split_idx, :]
validation = data.iloc[split_idx:, :]

# we have these data files but they're useless without labels
#validation = pd.read_csv('data_valid.csv', header=None, names=features)
#test = pd.read_csv('data_test.csv', header=None, names=features)

# labels we're trying to predict
concise_label = pd.read_csv('conciseness_train.labels', header=None)
clarity_label = pd.read_csv('clarity_train.labels', header=None)

# split them into train and validation data sets
concise_train = concise_label.iloc[:split_idx, :]
clarity_train = clarity_label.iloc[:split_idx, :]
concise_validation = concise_label.iloc[split_idx:, :]
clarity_validation = clarity_label.iloc[split_idx:, :]

# create features from text
# normally we'd use TD-IDF, but since titles are rather short, TD-IDF features
# are likely to be rather noisy
# instead, we'll use a binary CountVectorizer, which will make for more stable
# features (hopefully)
vectorizer = CountVectorizer(binary=True)
train_title_vecs = vectorizer.fit_transform(train['title'].values)
validation_title_vecs = vectorizer.transform(validation['title'].values)

# train an SVM binary classifier for each of these labels
# hyperparameter documentation found at:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
concise_cls = svm.SVC(kernel='rbf', C=1e3, gamma='auto', tol=1e-3, probability=True)
clarity_cls = svm.SVC(kernel='rbf', C=1e3, gamma='auto', tol=1e-3, probability=True)

print('training...')
concise_cls.fit(train_title_vecs, np.ravel(concise_train.values))
#clarity_cls.fit(title_vecs, clarity_train.values)

mse = mean_squared_error(np.ravel(concise_validation.values), concise_cls.predict(validation_title_vecs))
print(mse)