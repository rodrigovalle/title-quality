from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.models import  load_model
import keras
from sklearn.model_selection import StratifiedKFold
import spacy


from sklearn import preprocessing


def clean(df):
  df['description'].fillna("NULLD", inplace=True)
  df['category1'] = df['category1'].fillna("NULLC1")
  df['category2'] = df['category2'].fillna("NULLC2")
  df['category3'] = df['category3'].fillna("NULLC3")
  df['product_type'] = df['product_type'].fillna("NULLPT")
  df['title'] = [x.lower() for x in df['title']]
  df['description'] = tqdm([bs(x, "html5lib").text.lower() for x in df['description']])
  df['description'] = [x if len(x) > 0 else "NULLD" for x in df['description']]

def mean_embedding_vectorizer(X, w2v):

  dim = len(list(w2v.values())[0])


  return np.array([
    np.mean([w2v[w] for w in X if w in w2v] or [np.zeros(dim)], axis=0)])

def letter_embedding_vectorizer(X, w2v):
  dim = len(list(w2v.values())[0])

  return np.array([
    np.mean([w2v[w] for w in words if w in w2v] or [np.zeros(dim)], axis=0) for words in X])

def generate_features(df):

  data = []
  w2v = {}
  nlp = spacy.load('en')

  with open("data/glove.6B/glove.6B.200d.txt", "rb") as lines:
    for line in tqdm(lines):
      w2v[line.split()[0].decode("utf-8")] = np.array(list(map(float, line.split()[1:])))

  for i,row in tqdm(df.iterrows()):

    features = []

    title_words = row['title'].split()
    desc_words = str(row['description']).split()
    cat1_words = str(row['category1']).split()
    cat2_words = str(row['category2']).split()
    cat3_words = str(row['category3']).split()
    sku_id = row['sku_id']
    doc = nlp(row['title'])
    #generate part of speech features
    pos = []

    for token in doc:
      pos.append(token.pos_)
    features.append(pos.count("DET"))
    features.append(pos.count("ADJ"))
    features.append(pos.count("ADP"))
    features.append(pos.count("ADV"))
    features.append(pos.count("AUX"))
    features.append(pos.count("CONJ"))
    features.append(pos.count("INTJ"))
    features.append(pos.count("NOUN"))
    features.append(pos.count("NUM"))
    features.append(pos.count("PART"))
    features.append(pos.count("PRON"))
    features.append(pos.count("PROPN"))
    features.append(pos.count("PUNCT"))
    features.append(pos.count("SCONJ"))
    features.append(pos.count("SYM"))
    features.append(pos.count("VERB"))
    features.append(pos.count("X"))

    #generate noun chunk features
    noun_roots = []
    for chunk in doc.noun_chunks:
      noun_roots.append(chunk.root.text)

    noun_roots_embed = mean_embedding_vectorizer(noun_roots, w2v)[0]


    #generate basic features
    features.append(len(row['title']))
    features.append(len(title_words))
    features.append(len(title_words) - len(set(title_words)))
    features.append(len(set(title_words)))
    features.append(len([w for w in title_words if w.isdigit()]))
    features.append(len([c for c in row['title'] if c.isdigit()]))
    features.append(row['price'])


    #generate glove embedding features
    title_embed = mean_embedding_vectorizer(title_words, w2v)[0]

    for dim in title_embed:
      features.append(dim)

    if len(desc_words) > 0:
      desc_embed = mean_embedding_vectorizer(desc_words, w2v)[0]
      features.append(np.linalg.norm(title_embed - desc_embed))

      desc_roots = []
      doc = nlp(row["description"])
      for chunk in doc.noun_chunks:
        desc_roots.append(chunk.root.text)

      desc_roots_embed = mean_embedding_vectorizer(desc_roots, w2v)[0]

      features.append(np.linalg.norm(noun_roots_embed - desc_roots_embed))

    else:
      features.append(np.linalg.norm(title_embed))
      features.append(np.linalg.norm(noun_roots_embed))

    if len(cat1_words) > 0:
      cat1_embed = mean_embedding_vectorizer(cat1_words, w2v)[0]
      features.append(np.linalg.norm(title_embed - cat1_embed))

      cat1_roots = []
      doc = nlp(row["category1"])
      for chunk in doc.noun_chunks:
        cat1_roots.append(chunk.root.text)

      cat1_roots_embed = mean_embedding_vectorizer(cat1_roots, w2v)[0]

      features.append(np.linalg.norm(noun_roots_embed - cat1_roots_embed))
    else:
      features.append(np.linalg.norm(title_embed))
      features.append(np.linalg.norm(noun_roots_embed))

    if len(cat2_words) > 0:
      cat2_embed = mean_embedding_vectorizer(cat2_words, w2v)[0]
      features.append(np.linalg.norm(title_embed - cat2_embed))

      cat2_roots = []
      doc = nlp(row["category2"])
      for chunk in doc.noun_chunks:
        cat2_roots.append(chunk.root.text)

      cat2_roots_embed = mean_embedding_vectorizer(cat2_roots, w2v)[0]

      features.append(np.linalg.norm(noun_roots_embed - cat2_roots_embed))
    else:
      features.append(np.linalg.norm(title_embed))
      features.append(np.linalg.norm(noun_roots_embed))

    if len(cat1_words) > 0:
      cat3_embed = mean_embedding_vectorizer(cat3_words, w2v)[0]
      features.append(np.linalg.norm(title_embed - cat3_embed))

      cat3_roots = []
      doc = nlp(row["category3"])
      for chunk in doc.noun_chunks:
        cat3_roots.append(chunk.root.text)

      cat3_roots_embed = mean_embedding_vectorizer(cat3_roots, w2v)[0]

      features.append(np.linalg.norm(noun_roots_embed - cat3_roots_embed))
    else:
      features.append(np.linalg.norm(title_embed))
      features.append(np.linalg.norm(noun_roots_embed))

    data.append(features)

  return np.array(data)



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--nogeneration', '-n', action='store_true')
  parser.add_argument('--generationonly', '-g', action='store_true')
  parser.add_argument('--pretrained', '-p', action = 'store_true')


  args = parser.parse_args()

  #If p flag is set, simply load a pretrained model
  if args.pretrained:
    features = concise_features = np.load('data/concise_features.npy')
    df = pd.read_csv("data/data_train_clean.csv", dtype=object)
    X_train = concise_features
    y_train = df['conciseness'].values
    y_clarity_train = df['clarity'].values

    model = load_model('data/modelconcise.h5')

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

    cvscores_conciseness = []
    for train, test in kfold.split(X_train, y_train):
      y_train_cat = keras.utils.to_categorical(y_train, 2)
      loss_and_metrics = model.evaluate(X_train[test], y_train_cat[test], batch_size=128)
      print(loss_and_metrics)
      cvscores_conciseness.append(pow(loss_and_metrics[0], 0.5))

    model = load_model('data/modelclarity.h5')
    cvscores_clarity = []
    for train, test in kfold.split(X_train, y_clarity_train):
      y_train_cat = keras.utils.to_categorical(y_clarity_train, 2)
      loss_and_metrics = model.evaluate(X_train[test], y_train_cat[test], batch_size=128)
      print(loss_and_metrics)
      cvscores_clarity.append(pow(loss_and_metrics[0], 0.5))

    with open('./results/deeplearning', 'w') as f:
      f.write(("Results from pretrained DNN model:\n"))
      f.write("Average Conciseness score: {}\n".format( np.mean(cvscores_conciseness)))
      f.write("STD: {}\n".format(np.std(cvscores_conciseness)))
      f.write("Average Clarity score: {}\n".format(np.mean(cvscores_clarity)))
      f.write("Std: {}\n".format(np.std(cvscores_clarity)))


    exit(0)

  if args.nogeneration:
    df = pd.read_csv("data/data_train_clean.csv", dtype=object)

  else:
    #Add headers
    headers = ["country", "sku_id", "title", "category1", "category2", "category3", "description",
               "price", "product_type", "clarity", "conciseness"]

    #Read and preprocess training_data
    df = pd.read_csv("data_train.csv", delimiter=',', encoding='utf-8', names=headers)

    with open("conciseness_train.labels", 'r') as file:
      conciseness = [int(_.strip()) for _ in file.readlines()]
    df['conciseness'] = conciseness

    with open("clarity_train.labels", 'r') as file:
      clarity = [int(_.strip()) for _ in file.readlines()]
    df['clarity'] = clarity

    clean(df)
    df.to_csv('data/data_train_clean.csv', ',', encoding='utf-8', index=False)

  if args.nogeneration:
    concise_features = np.load('data/concise_features.npy')

  else:
    concise_features = generate_features(df)
    np.save('data/concise_features.npy', concise_features)

  if args.generationonly:
    print("Generation complete")
    exit()


  #Split out test set
  X_train = concise_features
  y_train = df['conciseness'].values
  y_clarity_train = df['clarity'].values

  shape = concise_features.shape

  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)


  #Train Conciseness Model
  cvscores_conciseness = []
  for train,test in kfold.split(X_train, y_train):
    print("train", train)
    print("test", test)

    y_train_cat = keras.utils.to_categorical(y_train, 2)
    #Build keras model
    model = Sequential()
    model.add(Dense(units=100, activation ='relu', input_dim=shape[1]))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))

    sgd = optimizers.Adagrad(lr=0.01)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mse'])

    #Train model
    model.fit(X_train[train], y_train_cat[train], epochs=150, batch_size=128, validation_data=(X_train[test], y_train_cat[test]))

    loss_and_metrics = model.evaluate(X_train[test], y_train_cat[test], batch_size=128)
    print(loss_and_metrics)
    cvscores_conciseness.append(pow(loss_and_metrics[0],0.5))

  model.save("data/modelconcise.h5")


  #Train Clarity Model
  cvscores_clarity = []
  for train,test in kfold.split(X_train, y_clarity_train):

    y_train_cat = keras.utils.to_categorical(y_clarity_train, 2)
    X_train = preprocessing.normalize(X_train,axis=1)
    #Build keras model
    model = Sequential()
    model.add(Dense(units=100, activation ='relu', input_dim=shape[1]))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))

    sgd = optimizers.Adagrad(lr=0.01)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mse'])

    #Train model
    model.fit(X_train[train], y_train_cat[train], epochs=100, batch_size=128, validation_data=(X_train[test], y_train_cat[test]))

    loss_and_metrics = model.evaluate(X_train[test], y_train_cat[test], batch_size=128)
    print(loss_and_metrics)
    cvscores_clarity.append(pow(loss_and_metrics[0],0.5))

  model.save("data/modelclarity.h5")

  #Evaluate
  print("final conciseness score", np.mean(cvscores_conciseness))
  print("std", np.std(cvscores_conciseness))
  print("final clarity score", np.mean(cvscores_clarity))
  print("std", np.std(cvscores_clarity))

