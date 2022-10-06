# train multinomial logistic regression model with word2vec
import os
import sys
import pandas as pd
import numpy as np
import re
import string
import pickle
import spacy
import argparse
import json
from typing import List, Dict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import multilabel_confusion_matrix

LEMMATIZER = WordNetLemmatizer()


def _preprocess(text: str) -> str:
    """
    Preprocess text (clean, lemmatize, stopwords)
    :param text:
    :return:
    """
    # lowercase
    text = text.lower()

    # substitute number for token
    text = re.sub(r'\b\d+\b', 'NUMBER', text)

    # substitute big spaces with little spaces
    text = re.sub(r'\s+', ' ', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # split into tokens
    tokens = text.split()

    # remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]

    # lemmatize
    tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens]

    return ' '.join(tokens)


def train(data: List[Dict], outdir: str, option: str = 'tfidf', num_folds: int = 5):
    """
    Train model
    :param data:
    :param outdir:
    :param option:
    :param num_folds:
    :return:
    """
    print(f'Multi-target random forest with {option} vectors and {num_folds} folds...')

    # get X and y
    X = [entry['text'] for entry in data]
    y = [entry['labels'] for entry in data]

    valid_inds = [ind for ind, text in enumerate(X) if text.strip()]
    X = [X[ind] for ind in valid_inds]
    y = [y[ind] for ind in valid_inds]
    print(f'{len(valid_inds)} valid sentences of {len(data)}')

    # compute vectors
    if option == 'tfidf':
        print('Computing tfidf vectors...')
        # preprocess text
        X = [_preprocess(entry) for entry in X]
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        X_vectors = tfidf_vectorizer.fit_transform(X)
    elif option == 'spacy':
        print('Getting spacy vectors...')
        # lowercase text
        X = [entry.lower() for entry in X]

        nlp = spacy.load("en_core_sci_sm")
        X_vectors = []
        for text in X:
            doc = nlp(text)
            X_vectors.append(doc.vector)
        X_vectors = np.stack(X_vectors, axis=0)
    else:
        raise NotImplementedError(f"Unknown option! {option}")

    print('Training...')

    # define the multinomial random forest
    forest = RandomForestClassifier(random_state=12345)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)

    # define the model evaluation procedure
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=12345)

    # evaluate the model and collect the scores
    n_scores_a = cross_val_score(multi_target_forest, X_vectors, y, scoring='accuracy', cv=cv, n_jobs=-1)
    n_scores_f1 = cross_val_score(multi_target_forest, X_vectors, y, scoring='f1_weighted', cv=cv, n_jobs=-1)

    # confusion matrix
    y_pred = cross_val_predict(multi_target_forest, X_vectors, y, cv=cv)
    confusion_matrix = multilabel_confusion_matrix(y, y_pred)
    print(confusion_matrix)

    # report the model performance
    print('Accuracy: ', n_scores_a)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores_a), np.std(n_scores_a)))
    print('F1 scores: ', n_scores_f1)
    print('Mean F1 (weighted): %.3f (%.3f)' % (np.mean(n_scores_f1), np.std(n_scores_f1)))

    # save model
    outfile = os.path.join(outdir, f'rf_{option}_{num_folds}.pkl')
    pickle.dump(multi_target_forest, open(outfile, 'wb'))
    print(f'Model saved to {outfile}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to prepped data file")
    parser.add_argument("--folds", type=int, help="Number of folds for cross-validation", default=5)
    parser.add_argument("--option", type=str, help="Vectorization options (tfidf, spacy)")
    parser.add_argument("--outdir", type=str, help="Output directory for model files")
    args = parser.parse_args()

    data_file = args.data
    n_folds = args.folds
    option = args.option
    outdir = args.outdir

    if not os.path.exists(data_file):
        print('Data file does not exist!')
        sys.exit(-1)

    if option not in {'tfidf', 'spacy'}:
        print('Unknown option!')
        sys.exit(-1)

    if not outdir:
        print('No output directory specified!')
        sys.exit(-1)

    os.makedirs(outdir, exist_ok=True)

    # read data
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)

    print(f'{len(data)} rows read')

    # train model
    train(data, outdir=outdir, option=option, num_folds=n_folds)

    print('done.')