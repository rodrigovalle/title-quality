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
