# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import pandas as pd



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
    
    
def split(data):
    """Split the data in 90% training and 10% testing
    data : pandas data frame of the training data set"""
    
    n_movies = data ['movie_id'].nunique()
    n_users = data ['user_id'].nunique()
    
    data_train = data.iloc[:int(data.shape[0]*0.9)]
    data_test = data.iloc[int(data.shape[0]*0.9):]
    
    return data_train, data_test,n_movies, n_users
