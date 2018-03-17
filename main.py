from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

from gensim.models import doc2vec
from gensim.utils import simple_preprocess

from helper import caching_trainer, load_data

from concurrent.futures import ProcessPoolExecutor, wait
import pandas as pd
import re, string
import math

alphanum = re.compile(r'[a-zA-Z0-9_]+')
words = re.compile(r'\w+')

def strip_alphanum(s):
    return alphanum.sub('', s)

def read_titles_tagged(X):
    for title, c1, c2, c3 in zip(X.title, X.category1, X.category2, X.category3):
        yield doc2vec.TaggedDocument(simple_preprocess(title), [c1, c2, c3])

# create features from text
def get_features(X, model=None):
    #vectorizer = CountVectorizer()
    #title_word_freq = vectorizer.fit_transform(X_train['title'].values)

    # train doc2vec
    if not model:
        model = doc2vec.Doc2Vec(vector_size=300, window=10, min_count=1, workers=4, epochs=100)
        model.build_vocab(read_titles_tagged(X))
        model.train(read_titles_tagged(X), total_examples=model.corpus_count, epochs=model.epochs)

    title_doc2vec = X.title.apply(lambda s: pd.Series(model.infer_vector(s)))
    title_char_len = X.title.apply(len) # ?
    title_word_len = X.title.apply(lambda s: len(words.findall(s)))
    title_nonalphanum = X.title.apply(lambda s: len(strip_alphanum(s)))

    return pd.concat(
        [
            title_doc2vec,    # 300
            title_char_len,   # 1
            title_word_len,   # 1
            title_nonalphanum # 1
        ],
        axis=1
    ).values, model

# train an SVM binary classifier for each of these labels
# hyperparameter documentation found at:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# def train_svm()
#     concise_cls = svm.SVC(kernel='rbf', tol=1e-2, probability=True) # class_weight='balanced'
#     clarity_cls = svm.SVC(kernel='rbf', tol=1e-2, probability=True) # class_weight='balanced'
# 
#     #concise_fit = lambda: concise_cls.fit(X_train, concise_train)
#     #clarity_fit = lambda: clarity_cls.fit(X_train, clarity_train)
# 
#     #concise_cls = caching_trainer(concise_fit, 'concise_svm')
#     #clarity_cls = caching_trainer(clarity_fit, 'clarity_svm')

def grid_search(X, y, svm, filename):
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
    smote_pipeline = Pipeline([('smote', smote_enn), ('svm', svm)])

    cv = StratifiedShuffleSplit(n_splits=4, train_size=0.2, test_size=0.2)
    gridsearch = GridSearchCV(smote_pipeline, parameters, cv=cv, scoring=scorer,
                              n_jobs=4, verbose=10, refit=False)

    gridsearch = caching_trainer(gridsearch, X, y, filename)

#y_pred = concise_cls.predict_proba(X_test)[:,1]
#y_true = concise_test

def train(estimator, X, y, filename):
    print('training ' + filename)
    return caching_trainer(estimator, X, y.values.ravel(), filename, retrain=False)

if __name__ == '__main__':
    # load and unpack data
    dataset = load_data()
    clarity_train_X, clarity_train_y = dataset['clarity_train']
    clarity_test_X, clarity_test_y = dataset['clarity_test']
    concise_train_X, concise_train_y = dataset['concise_train']

    clarity_cls = svm.SVC(kernel='rbf', C=1e5, gamma=1e-1, tol=1e-2, probability=True, verbose=True, cache_size=1000) # class_weight='balanced'
    concise_cls = svm.SVC(kernel='rbf', C=1e5, gamma=1e-1, tol=1e-2, probability=True, verbose=True, cache_size=6000) # class_weight='balanced'

    with ProcessPoolExecutor(max_workers=2) as executor:
        clarity_feature_fut = executor.submit(get_features, clarity_train_X)
        concise_feature_fut = executor.submit(get_features, concise_train_X)

        clarity_features, clarity_doc2vec = clarity_feature_fut.result()
        concise_features, concise_doc2vec = concise_feature_fut.result()

        # shuffle split for concise SVM because it takes too long on full data
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.2, test_size=0.2, random_state=1)
        train_idx, test_idx = next(sss.split(concise_features, concise_train_y))
        concise_features_sss = concise_features[train_idx]
        concise_train_y_sss = concise_train_y.iloc[train_idx]
        concise_features_test_sss = concise_features[test_idx]
        concise_test_y_sss = concise_train_y.iloc[test_idx]

        f1 = executor.submit(train, clarity_cls, clarity_features, clarity_train_y, 'clarity_svm2')
        f2 = executor.submit(train, concise_cls, concise_features_sss, concise_train_y_sss, 'concise_svm2')

        clarity_cls = f1.result()
        concise_cls = f2.result()

        clarity_test_features, _ = get_features(clarity_test_X, model=clarity_doc2vec)

        clarity_y_pred = clarity_cls.predict_proba(clarity_test_features)[:, 1]
        concise_y_pred = concise_cls.predict_proba(concise_features_test_sss)[:, 1]

        print('clarity rmse:', math.sqrt(mean_squared_error(clarity_test_y, clarity_y_pred)))
        print('concise rmse:', math.sqrt(mean_squared_error(concise_test_y_sss, concise_y_pred)))
