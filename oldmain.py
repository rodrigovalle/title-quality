import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
import re
import os
import shutil


logging.getLogger().setLevel(logging.INFO)
WORDS_FEATURE = 'words'
EMBEDDING_SIZE = 500
n_words = 0
MAX_LABEL = 2
HIDDEN_UNITS = [150, 200]

features = ['country', 'sku_id', 'title', 'category1', 'category2', 'category3',
            'description', 'price', 'product_type']

# training, test, and validation data
train = pd.read_csv('data_train.csv', header=None, names=features)
test = pd.read_csv('data_test.csv', header=None, names=features)
validation = pd.read_csv('data_valid.csv', header=None, names=features)

# labels we're trying to predict
concise_train = pd.read_csv('conciseness_train.labels', header=None)
clarity_train = pd.read_csv('clarity_train.labels', header=None)


vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(1000)
category2_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(1000)

descriptions = [re.sub("<.*?>", " ", str(x)) for x in list(train["description"])]

vectorizer = CountVectorizer(binary=True)

train_title_vecs = vectorizer.fit_transform(train['title'].values)
train_title_vecs = train_title_vecs.todense()

cat1 = [str(x) for x in train["category1"].values]
cat2 = [str(x) for x in train["category2"].values]

X_transform_train = vocab_processor.fit_transform(cat1)
X_cat2_transform = category2_processor.fit_transform(cat2)

X_titlelengths = [[len(x)] for x in train["title"].values]
X_desciptionlengths = [[len(str(x))] for x in train["description"].values]


X_titlelengths = np.asarray(X_titlelengths)
X_desciptionlengths = np.asarray(X_desciptionlengths)



X_train = np.array(list(X_transform_train))
X_cat2 = np.array(list(X_cat2_transform))


#X_train = np.array(X_titlelengths)

#X, X_test, y, y_test = train_test_split(X_train, concise_train.values, test_size = 0.1, random_state=1)

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=1)

train_idx = 30000

X_test = X_train[30000:]
X_train = X_train[:30000]

X_cat2_test = X_cat2[30000:]
X_cat2 = X_cat2[:30000]

X_titlelengths_test = X_titlelengths[30000:]
X_titlelengths = X_titlelengths[:30000]

X_desciptionlengths_test = X_desciptionlengths[30000:]
X_desciptionlengths = X_desciptionlengths[:30000]

y_test = concise_train.values[30000:]
y_train = concise_train.values[:30000]


print(X_train.shape)

n_words = len(vocab_processor.vocabulary_)
cat2_words = len(category2_processor.vocabulary_)

feature_dict = {
  "t_len": X_titlelengths,
  WORDS_FEATURE: X_train,

}

feature_dict_test = {
  "t_len": X_titlelengths_test,
  WORDS_FEATURE: X_test,

}

feature_columns = []
for key in feature_dict.keys():
  if key != WORDS_FEATURE and key != "cat2":
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# TODO: convert each row to a numerical vector
# (use sklearn.feature_extraction)


def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def rnn_model(features, labels, mode):
  word_vectors = tf.contrib.layers.embed_sequence(features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis = 1)

  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

  _, encoding = tf.nn.static_rnn(cell, word_list, dtype = tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(logits=logits, labels=labels, mode=mode)

def bag_of_words_model(features, labels, mode):
  """A bag-of-words model. Note it disregards the word order in the text."""


  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)

  '''
  cat2_column = tf.feature_column.categorical_column_with_identity(
    "cat2", num_buckets=cat2_words)
  cat2_embedding_column = tf.feature_column.embedding_column(
    cat2_column, dimension=EMBEDDING_SIZE)
  '''

  bow = tf.feature_column.input_layer(
      features,
    feature_columns= [bow_embedding_column] + feature_columns)




  #bow = tf.reshape(features[WORDS_FEATURE], [1, n_words + 1, 1])
  #bow = tf.feature_column.input_layer(features, my_feature_columns)
  print("SHAPE OF BOW{}".format(bow.get_shape()))

  for units in HIDDEN_UNITS:
    bow = tf.layers.dense(bow, units=units, activation=tf.nn.relu)

  dropout = tf.layers.dropout(inputs=bow, rate = 0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)

#Build Model



X_train -= 1
X_test -= 1

model_fn = bag_of_words_model

model_dir = "./model"

classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir= model_dir)

print(y_train.shape)

train_input_fn = tf.estimator.inputs.numpy_input_fn(feature_dict, y=y_train, batch_size=128, num_epochs=100, shuffle=True)

classifier.train(input_fn=train_input_fn)

#Predict
test_input_fn = tf.estimator.inputs.numpy_input_fn(feature_dict_test, y=y_test, num_epochs=1, shuffle=False)

predictions = classifier.predict(input_fn=test_input_fn)

y_predicted = np.array(list(p['prob'][p['class']] for p in predictions))
#y_predicted = np.array(list(p['class']for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test).shape)

#print(list(y_predicted))
score = pow(mean_squared_error(y_test, y_predicted), 0.5)

acc = metrics.accuracy_score(y_test, np.around(y_predicted))

print("RMSE, acc: {}, {}".format(score, acc))

shutil.rmtree(model_dir)










