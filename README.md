## Team Members
Yuhao (Bill) Tang 104621566\
Rodrigo Valle 104494120\
Krit Saefang 904723127

## Setting up a development environment
Make sure you have Python 3.6 installed, and run the following commands:

```bash
$ python -m venv venv/
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Training data
**data_train.csv** is a CSV text file containing product features in the
following format:

```csv
country, sku_id, title, category_lvl_1, category_lvl_2, category_lvl_3,
description, price, product_type
```

**clarity_train.labels** is a text file that contains ground truth labels for
the Clarity scoring. Each line corresponds to the same product in
*data_train.csv*. The label has two possible values: 1 = clear or 0 = not clear.
Distribution in this file: 94.3% are labeled 1, while 5.7% are labeled 0.

**conciseness_train.labels** is a text file that contains ground truth labels
for the Conciseness scoring. Each line corresponds to the same product in
*data_train.csv*. The label has two possible values: 1 = concise or 0 = not
concise. Distribution in this file: 68.5% are labeled 1, while 31.7% are labeled
0.

## Validation Data
**data_valid.csv** is a text file containing product features. It has the same
format as *data_train.csv*.

## Testing Data
**data_test.csv** is a text file containing product features. It has the same
format as *data_train.csv*.

## INSTRUCTIONS FOR TAS

Dependencies in requirements.txt must be installed to run the **run.sh** script

```bash
$ ./run.sh
```

Results for the deep neural net, naive bayes, and svm models will be output to 
**deeplearning.txt**, **naivebayes.txt**, and **svm.txt** respectively in the results
folder. 
    

Due to the extremely long training time required to train the SVM model, by default
our script use pretrained models to predict results. Details on how to train a model from 
scratch can be found in the Running SVM Model Section

Also, to reduce the time required, we have included pregenerated features required for the 
deep neural net model to train. For instructions on how to generate features from scratch, 
view the Running Deep Leaning Model section. 

By default, our run.sh will train a new neural net using pregenerated feautures,
but we have included instructions on how
to simply run predictions using a pretrained model in case you do not wish to wait
for a new model to train. 

## Running Deep Learning model
**deeplearning.py** contains the code for feature generation and training of our
DNN model, built using Keras on top of Tensorflow.

For the word embeddings, you must first download the GloVe 6B pretrained word vectors
from the official glove project site [here](http://nlp.stanford.edu/data/glove.6B.zip)
and place the unzipped folder into the data folder.

This model also requires Tensorflow, Keras, Sci-kit learn, tqdm, BeautifulSoup4,
numpy, pandas, and Spacy.

After installing Spacy, the en language model must be downloaded with
```bash
$ python -m spacy download en
```
before features can be generated. 

To run feature generation and training + model evaluation simply run:

```bash
$ python deeplearning.py
```
Note: Feature generation + cross validation training can take a great deal of time

If you only want to generate cleaned data and features, use the -g flag.
This should take around 25 minutes

If you want to use the pregernerated cleaned data and features included, use the -n flag
to proceed directly to model training.
This should take around 30-45 minutes for the full cross validation of both models

To test the model on the pretrained models included in the data folder, use the -p flag
This should take about 10 seconds

## Running SVM Model

```bash
$ python main.py
```

If you wish to retrain the models from scratch instead of loading the pretrained
files, be warned that this will take upwards of 8 hours (parallelization not
possible):

Change line 95 in the top-level train function: set `retrain=True`.

#Running Naive Bayes Model
## Running
To run the implementation, simply run the command with arguments a through f:

```bash
$ python main_NB.py a b c d e f
```

the arguments are as follows:

**a**: 0 for clarity, 1 for concise, 2 for pre-generated features (Requires 'concise_features.npy' in Data folder)

**b**: 0 for GaussianNB, 1 for MultinomialNB, 2 for BernoulliNB

**c**: 0 for no smoothing, 1 for smoothing

**d**: 0 for title/description only as features, 1 for using categories as well

**e**: 0 for not using fit priors, 1 for using fit priors

**f**: 0 for not using class priors, 1 for using class priors

All arguments must be given

