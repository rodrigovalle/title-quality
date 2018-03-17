#!/bin/bash



echo "Running SVM"
python3 main.py > results/svm.txt

touch results/naivebayes.txt
echo "Running Gaussian Naive Bayes"
echo "Results will be recorded to results/naivebayes.txt"
echo "GaussianNB Results" >> results/naivebayes.txt

python3 main_NB.py 0 0 1 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 0 1 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 0 1 0 0 1 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 2 0 1 0 0 0 >> results/naivebayes.txt 2> /dev/null

echo "Running Multinomial Naive Bayes"
echo "MultinomialNB Results" >> results/naivebayes.txt

python3 main_NB.py 0 1 1 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 1 0 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 1 1 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 1 0 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 1 1 0 1 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 1 1 0 0 1 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 1 1 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 1 0 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 1 1 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 1 0 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 1 1 0 1 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 1 1 0 0 1 >> results/naivebayes.txt 2> /dev/null

echo "Running Bernoulli Naive Bayes"
echo "BernoulliNB Results" >> results/naivebayes.txt
python3 main_NB.py 0 2 1 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 2 0 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 2 1 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 2 0 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 2 1 0 1 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 0 2 1 0 0 1 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 2 1 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 2 0 0 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 2 1 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 2 0 1 0 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 2 1 0 1 0 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 1 2 1 0 0 1 >> results/naivebayes.txt 2> /dev/null

python3 main_NB.py 2 2 1 0 0 0 >> results/naivebayes.txt 2> /dev/null
echo "Running Deep Neural Network"
echo "Results will be written to results/deeplearning.txt"
python3 deeplearning.py -n
echo "Script Completed"
