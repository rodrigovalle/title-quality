from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

from helper import caching_trainer, load_data

# load and unpack data
dataset = load_data()
train, test = dataset['data']
concise_train, concise_test = dataset['concise_label']
clarity_train, clarity_test = dataset['clarity_label']

# create features from text
# normally we'd use TD-IDF, but since titles are rather short, TD-IDF features
# are likely to be rather noisy
# instead, we'll use a binary CountVectorizer, which will make for more stable
# features (hopefully)
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(train['title'].values)
X_test = vectorizer.transform(test['title'].values)

# train an SVM binary classifier for each of these labels
# hyperparameter documentation found at:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
concise_cls = svm.SVC(kernel='rbf', tol=1e-3, probability=True)
clarity_cls = svm.SVC(kernel='rbf', tol=1e-3, probability=True)

#concise_fit = lambda: concise_cls.fit(X_train, concise_train)
#clarity_fit = lambda: clarity_cls.fit(X_train, clarity_train)

#concise_cls = caching_trainer(concise_fit, 'concise_svm')
#clarity_cls = caching_trainer(clarity_fit, 'clarity_svm')

parameters = {
    'C': [10**i for i in range(-5, 16, 2)],
    'gamma': [10**i for i in range(-15, 3, 2)]
}

def proba_mse(y, y_pred):
    # get just the predictions for concise/clarity = 1
    y_pred = y_pred[:, 1]
    return mean_squared_error(y, y_pred)

scorer = make_scorer(proba_mse, needs_proba=True, greater_is_better=False)

print('starting grid search...', flush=True)
gridsearch = GridSearchCV(concise_cls, parameters, scoring=scorer)
gridsearch_fit = lambda: gridsearch.fit(X_train, concise_train)

caching_trainer(gridsearch_fit, 'gridsearch_concise')

print(gridsearch_fit.cv_results_)

#y_pred = concise_cls.predict_proba(X_test)[:,1]
#y_true = concise_test

