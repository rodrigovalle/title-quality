#!/bin/bash
echo "Results will be recorded to NBResults.txt"
touch NBResults.txt
echo "Running Gaussian Naive Bayes"
echo "GaussianNB Results" >> NBResults.txt
echo "\n" >> NBResults.txt
python3 main_NB.py 0 0 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 0 1 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 0 1 0 0 1 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 2 0 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
echo "Running Multinomial Naive Bayes"
echo "MultinomialNB Results" >> NBResults.txt
echo "\n" >> NBResults.txt
python3 main_NB.py 0 1 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 1 0 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 1 1 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 1 0 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 1 1 0 1 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 1 1 0 0 1 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 1 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 1 0 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 1 1 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 1 0 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 1 1 0 1 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 1 1 0 0 1 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
echo "Running Bernoulli Naive Bayes"
echo "BernoulliNB Results" >> NBResults.txt
python3 main_NB.py 0 2 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 2 0 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 2 1 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 2 0 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 2 1 0 1 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 0 2 1 0 0 1 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 2 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 2 0 0 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 2 1 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 2 0 1 0 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 2 1 0 1 0 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 1 2 1 0 0 1 >> NBResults.txt 2> /dev/null
echo "\n" >> NBResults.txt
python3 main_NB.py 2 2 1 0 0 0 >> NBResults.txt 2> /dev/null
echo "Script Completed"
