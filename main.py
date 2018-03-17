from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

from gensim.models import doc2vec

from helper import caching_trainer, load_data

import re, string

# load and unpack data
dataset = load_data()
clarity_train_X, clarity_train_y = dataset['clarity_train']
concise_train_X, concise_train_y = dataset['concise_train']

X_train = clarity_train_X
y_train = clarity_train_y

# create features from text
# normally we'd use TD-IDF, but since titles are rather short, TD-IDF features
# are likely to be rather noisy
# instead, we'll use a binary CountVectorizer, which will make for more stable
# features (hopefully)
alphanum = re.compile('[a-zA-Z0-9_]+')
words = re.compile('\w+')

def strip_alphanum(s):
    return alphanum.sub('', s)

#vectorizer = CountVectorizer()
#title_word_freq = vectorizer.fit_transform(X_train['title'].values)
title_char_len = X_train['title'].apply(len) # ?
title_word_len = X_train['title'].apply(lambda s: len(words.findall(s)))
title_nonalphanum = X_train['title'].apply(lambda s: len(strip_alphanum(s)))

# train an SVM binary classifier for each of these labels
# hyperparameter documentation found at:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
concise_cls = svm.SVC(kernel='rbf', tol=1e-2, probability=True) # class_weight='balanced'
clarity_cls = svm.SVC(kernel='rbf', tol=1e-2, probability=True) # class_weight='balanced'

#concise_fit = lambda: concise_cls.fit(X_train, concise_train)
#clarity_fit = lambda: clarity_cls.fit(X_train, clarity_train)

#concise_cls = caching_trainer(concise_fit, 'concise_svm')
#clarity_cls = caching_trainer(clarity_fit, 'clarity_svm')

parameters = {
    'svm__C': [10**i for i in range(-5, 16, 2)],
    'svm__gamma': [10**i for i in range(-15, 4, 2)]
}

def proba_mse(y, y_pred):
    # get just the predictions for concise/clarity = 1
    y_pred = y_pred[:, 1]
    return mean_squared_error(y, y_pred)

scorer = make_scorer(proba_mse, needs_proba=True, greater_is_better=False)

smote_enn = SMOTEENN()
smote_pipeline = Pipeline([('smote', smote_enn), ('svm', clarity_cls)])

cv = StratifiedShuffleSplit(n_splits=4, train_size=0.2, test_size=0.2)
gridsearch = GridSearchCV(smote_pipeline, parameters, cv=cv, scoring=scorer,
                          n_jobs=4, verbose=10, refit=False)
gridsearch_fit = lambda: gridsearch.fit(X_train, clarity_train_y)
gridsearch = caching_trainer(gridsearch_fit, 'gridsearch_clarity2')

#y_pred = concise_cls.predict_proba(X_test)[:,1]
#y_true = concise_test
