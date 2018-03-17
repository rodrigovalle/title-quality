import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
import sys
import math

# Flags for different tasks
isConcise = 0
isGauss = 0
isNomial = 0
isBern = 0
useCategories = 0
selfPrior = 1
useExtPrior = 0
conciseExtPrior = [0.685,0.315]
clarityExtPrior = [0.943,0.053]
extPrior = clarityExtPrior
alphavalue = 1

# Use premade features
useFeatures = 0

# Get arguments for operation
cmdargs = sys.argv
#print(cmdargs)
# First command is for determining

# Argument 1 determines concise or clarity. 0 for concise
if float(cmdargs[1]) == 0:
	isConcise = 1
# Argument 1 can also indicate usage of preset features
if float(cmdargs[1]) == 2:
	useFeatures = 1
	isConcise = 1
# Argument 2 determines NB type. 0 for gauss, 1 for nomial, 2 for bern
if float(cmdargs[2]) == 0:
	isGauss = 1
if float(cmdargs[2]) == 1:
	isNomial = 1	
if float(cmdargs[2]) == 2:
	isBern = 1
# Argument 3 determines smoothing. 0 for off.
if float(cmdargs[3]) == 0:
	alphavalue = 0
# Argument 4 determines category usage. 0 for off
if float(cmdargs[4]) != 0:
	useCategories = 1
# Argument 5 for fit prior. 0 for off
if float(cmdargs[5]) == 0:
	selfPrior = 0
# Argument 6 for class prior. 0 for off
if float(cmdargs[6]) != 0:
	useExtPrior = 1;


if isConcise:
	extPrior = conciseExtPrior

features = ['country', 'sku_id', 'title', 'category1', 'category2', 'category3',
            'description', 'price', 'product_type']

if isGauss:
	print('Using Gaussian Naive Bayes')
if isNomial:
	print('Using Multinomial Naive Bayes')
if isBern:
	print('Using Bernoulli Naive Bayes')

if isConcise:
	print('Conciseness Prediction')
	selectedFeatures = ['title']
	if useCategories:
		print('Adding Categories to Features')
		selectedFeatures = ['title', 'category1', 'category2', 'category3']
	if useFeatures:
		print('Using Pre-generated Features')
else:
	print('Clarity Prediction')
	selectedFeatures = ['description']
	if useCategories:
		print('Adding Categories to Features')
		selectedFeatures = ['description', 'category1', 'category2', 'category3']

# training, test, and validation data
if useFeatures:
	data = np.load('data/concise_features.npy')
	split_idx = len(data) - 11000 # amount to split for validation data
	trainData = data[:split_idx, :]
	#print('Test Data length:')
	#print(len(trainData))
	validationData = data[split_idx:, :]
	#print('ValidationData length:')
	#print(len(validationData))
else:
	data = pd.read_csv('data_train.csv', header=None, names=features)
	split_idx = len(data) - 11000 # amount to split for validation data

	data.drop(columns=['sku_id'], inplace=True)

	trainData = data.iloc[:split_idx, :]
	#print('Test Data length:')
	#print(len(trainData))
	validationData = data.iloc[split_idx:, :]
	#print('Validation Data length:')
	#print(len(validationData))

# labels we're trying to predict

if isConcise:
	label = pd.read_csv('conciseness_train.labels', header=None)
else:
	label = pd.read_csv('clarity_train.labels', header=None)

# split them into train and validation data sets
train = label.iloc[:split_idx, :]
#print('Length of training labels:')
#print(len(train))
validation = label.iloc[split_idx:, :]
#print('Length of validation labels:')
#print(len(validation))

# create features from text
# normally we'd use TD-IDF, but since titles are rather short, TD-IDF features
# are likely to be rather noisy
# instead, we'll use a binary CountVectorizer, which will make for more stable
# features (hopefully). Bernoulli will require HashingVectorizer to save memory
# requirements
if isBern:
	vectorizer = HashingVectorizer(binary=True)
else:
	vectorizer = CountVectorizer(binary=True)

if isGauss:
	if useFeatures:
		#print('Setting Pre-generated Features')
		train_vecs = trainData
		validation_vecs = validationData
	else:
		#print('Training Vectors Set - Dense')
		train_vecs = vectorizer.fit_transform([str(x) for x in trainData[selectedFeatures].values]).todense()
		validation_vecs = vectorizer.transform([str(x) for x in validationData[selectedFeatures].values]).todense()	
else:
	if useFeatures:
		#print('Setting Pre-generated Features')
		train_vecs = trainData
		validation_vecs = validationData
	else:
		#print('Training Vectors Set')
		train_vecs = vectorizer.fit_transform([str(x) for x in trainData[selectedFeatures].values])
		validation_vecs = vectorizer.transform([str(x) for x in validationData[selectedFeatures].values])


# train an SVM binary classifier for each of these labels
# hyperparameter documentation found at:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
if isNomial:
	#print('Predictor set to Multinomial')
	if useExtPrior:
		print('Using Class Priors')
		predictor = MultinomialNB(alpha=alphavalue, fit_prior=selfPrior, class_prior=extPrior)
	else:
		predictor = MultinomialNB(alpha=alphavalue, fit_prior=selfPrior)
if isGauss:
	#print('Predictor set to Gaussian')
	if useExtPrior:
		print('Using Class Priors')
		predictor = GaussianNB(priors=extPrior)
	else:
		predictor = GaussianNB()
if isBern:
	#print('Predictor set to Bernoulli')
	if useExtPrior:
		print('Using Class Priors')
		predictor = BernoulliNB(alpha=alphavalue, fit_prior=selfPrior,  class_prior=extPrior)
	else:
		predictor = BernoulliNB(alpha=alphavalue, fit_prior=selfPrior)

#print('Training...')
predictor.fit(train_vecs, np.ravel(train.values))

#print('Predicting...')
y_pred = predictor.predict_proba(validation_vecs)[:,1]
y_true = np.ravel(validation.values)
mse = mean_squared_error(y_true, y_pred)
#print('MSE Results...')
#print(mse)
print('RMSE Results...')
print(math.sqrt(mse))


