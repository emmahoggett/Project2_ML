# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers import Reshape


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    ratings= pd.read_csv(path_dataset)
    split = ratings.Id.astype(str).str.split(r"r|_c", expand=True)
    prediction = ratings[['Prediction']]
    ratings.drop(['Id','Prediction'],axis = 1, inplace = True)
    ratings['user_id'] = split[1]
    ratings['movie_id'] = split[2]
    ratings['rating'] = prediction
    return ratings

def create_csv(path_dataset, submission):
    """Write results in csv format"""
    pos = ["r"+submission.user_id[i]+"_c"+submission.movie_id[i] for i in range(submission.shape[0])]
    result = pd.DataFrame({'Id':pos, 'Prediction':submission.rating})
    result = result.astype({'Id': str,'Prediction':int})
    return result.to_csv(path_dataset, index=False)

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x


def setDataSet(data):
    """Set global data set and split the data"""
    n_movies = data ['movie'].nunique()
    n_users = data ['user'].nunique()
    
    X = data[['user', 'movie']].values
    y = data ['rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
    X_train_array = [X_train[:,0], X_train[:,1]]
    X_test_array = [X_test[:,0], X_test[:,1]]
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder_train = encoder.transform(y_train)
    encoder_test = encoder.transform(y_test)

    y_train = np_utils.to_categorical(encoder_train)
    y_test = np_utils.to_categorical(encoder_test)
    
    return X_train_array, X_test_array, y_train, y_test, n_movies, n_users
