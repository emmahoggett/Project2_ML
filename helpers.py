# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pandas as pd


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


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
    pos = ["r"+submission.user_id[i]+"_c"+submission.movie_id[i] for i in range(submission.index.stop)]
    result = pd.DataFrame({'Id':pos, 'Prediction':submission.rating})
    result = result.astype({'Id': str,'Prediction':int})
    return result.to_csv(path_dataset, index=False)

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

def convert_train(ratings):
    """convert sparse rating matrix into a data frame"""
    #convert ratings into an array
    matrix_0 = ratings.toarray()
    matrix_1 = matrix_0.flatten()[matrix_0.flatten().nonzero()]
    
    #concatenate user_id, movie_id and ratings
    matrix_2 = np.vstack((matrix_0.nonzero()[0],matrix_0.nonzero()[1],matrix_1))
    matrix_3 = matrix_2.T.astype(int)
    
    # convert rating into a data frame
    ratings = pd.DataFrame({'user_id': matrix_3[:,0], 'movie_id': matrix_3[:,1],'rating':matrix_3[:,2]})
    n_users = len(ratings.user_id.unique())
    n_movies = len(ratings.movie_id.unique())
    
    return ratings, n_users, n_movies

