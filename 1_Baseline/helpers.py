# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pandas as pd
import csv

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines() 
    
    
def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)

def statistics(data):
    row = set([line[0] for line in data])
    col = set([line[1] for line in data])
    return min(row), max(row), min(col), max(col)
    
def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of users : {}, number of items: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings
    

# def create_csv(path_dataset, submission):
#     """Write results in csv format"""
#     result = pd.DataFrame({'Id':pos, 'Prediction':submission.rating})
#     result = result.astype({'Id': str,'Prediction':int})
#     return result.to_csv(path_dataset, index=False)

def create_csv(data, pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        pos = ["r"+str(data[i][0])+"_c"+str(data[i][1]) for i in range(len(data))]
        for r1, r2 in zip(pos, pred):
            writer.writerow({'Id':r1,'Prediction':int(r2)})


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

