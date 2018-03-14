from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

from helper import caching_trainer, load_data

# load and unpack data
dataset = load_data()
clarity_train_X, clarity_train_y = dataset['clarity_train']
concise_train_X, concise_train_y = dataset['concise_train']

# create features from text
# normally we'd use TD-IDF, but since titles are rather short, TD-IDF features
# are likely to be rather noisy
# instead, we'll use a binary CountVectorizer, which will make for more stable
# features (hopefully)
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(clarity_train_X['title'].values)

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

