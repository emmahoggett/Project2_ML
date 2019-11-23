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
    """Set embedding layer class"""
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x
    
    
def split(data):
    """Split the data"""
    
    n_movies = data ['movie_id'].nunique()
    n_users = data ['user_id'].nunique()
    
    data_train = data.iloc[:int(data.shape[0]*0.8)]
    data_test = data.iloc[int(data.shape[0]*0.8):]
    
    return data_train, data_test,n_movies, n_users

def splitClean(data):
    """Set global data set and split the data"""
    
    data_train, data_test,n_movies, n_users = split(data)
    
    f = ['count','mean']

    df_movie_summary = data_train.groupby('movie_id')['rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.95),0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index


    df_cust_summary = data_train.groupby('user_id')['rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.95),0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index
    
    data_train = data_train[~data_train['movie_id'].isin(drop_movie_list)]
    data_train = data_train[~data_train['user_id'].isin(drop_cust_list)]
    
    return data_train, data_test,n_movies, n_users

    
def setDataSet(data):
    """Split and clean the data set. Ratings are enconded and the data frame is 
    change into an array with movies and users"""
    
    data_train, data_test,n_movies, n_users = splitClean(data)
    user_enc = LabelEncoder()
    data_train ['user'] = user_enc.fit_transform(data_train['user_id'].values)
    data_test ['user'] = user_enc.fit_transform(data_test['user_id'].values)

    item_enc = LabelEncoder()
    data_train ['movie'] = item_enc.fit_transform(data_train['movie_id'].values)
    data_test ['movie'] = item_enc.fit_transform(data_test['movie_id'].values)

    data_train ['rating'] = data_train ['rating'].values.astype(np.int)
    data_test ['rating'] = data_test ['rating'].values.astype(np.int)
    
    X_train = data_train[['user', 'movie']].values
    y_train = data_train ['rating']
    
    X_test = data_test[['user', 'movie']].values
    y_test = data_test ['rating']
    
    X_train_array = [X_train[:,0], X_train[:,1]]
    X_test_array = [X_test[:,0], X_test[:,1]]
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder_train = encoder.transform(y_train)
    encoder_test = encoder.transform(y_test)

    y_train = np_utils.to_categorical(encoder_train)
    y_test = np_utils.to_categorical(encoder_test)
    
    return X_train_array, X_test_array, y_train, y_test, n_movies, n_users
