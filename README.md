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


## Deep Learning model
**deeplearning.py** contains the code for feature generation and training of our
DNN model, built using Keras on top of Tensorflow.

For the word embeddings, you must first download the GloVe 6B pretrained word vectors
from the official glove project site [here](http://nlp.stanford.edu/data/glove.6B.zip)
and place the unzipped folder into the data folder

To run feature generation and training + model evaluation simply run:

```bash
$ python deeplearning.py
```

If you only want to generate cleaned data and features, use the -g flag. If you already
have generated features available, use the -n flag to skip generation and go straight to
model training.



