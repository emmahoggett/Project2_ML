import tensorflow as tf
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import pandas as pd
from itertools import groupby

from helpersBaseline import *

def computeBaseline(train, test):
    """ Implementation of a baseline methods : global mean, user mean, item mean,
        and matrix factorisation using SGD as well as ALS.
    
        train : trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the columns 'global_mean_rating' 
        'user_mean_rating', 'item_mean_rating', 'MF_SGD_rating', 'MF_ALS_rating'.
    
    """
    
    print ("Starting to compute global mean...")
    global_mean = computeGlobalMean(train, test)
    print ("... Finished sucessfully")
    
    print ("Starting to compute user mean...")
    user_mean = computeUserMean(train, test)
    print ("... Finished sucessfully")
    
    print ("Starting to compute item mean...")
    item_mean = computeItemMean(train, test)
    print ("... Finished sucessfully")

    print ("Starting to compute MF using ALS...")
    MF_ALS = computeMFALS(train, test)
    print ("... Finished sucessfully")    
    
    mean_rating = global_mean.merge(user_mean, on=['user_id', 'movie_id'])
    mean_rating = mean_rating.merge(item_mean, on=['user_id', 'movie_id'])
    mean_rating = mean_rating.merge(MF_ALS, on=['user_id', 'movie_id'])
    
    return mean_rating

###################################################

def computeGlobalMean(train, test):
    """
    Implementation of global mean.
    
        train: train set build to fit the model.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'global_mean_rating'.
    
    """
    df = test[['user_id', 'movie_id']].copy()
    
    global_mean = train[['rating']].mean()
    df['global_mean_rating'] = np.clip(df[['user_id', 'movie_id']].apply(lambda row: global_mean, axis=1), 1, 5)
    
    return df

def computeUserMean(train, test):
    """
    Implementation of user mean.
    
        train: trainset build to fit the model.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'user_mean_rating'.
    
    """ 
    data = preprocess_data(train) # transform dataframe into sparse matrix
    
    num_users, num_items = data.shape
    user_mean = np.zeros((num_users,1))

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = data[user_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
            user_mean[user_index] = user_train_mean
    
    df = test[['user_id', 'movie_id']].copy()
    
    df['user_mean_rating'] = np.clip(user_mean[test['user_id'].values.astype(int)-1], 1, 5)
    
    return df


def computeItemMean(train, test):
    """
    Implementation of item mean.
    
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'item_mean_rating'.
    
    """
    data = preprocess_data(train) # transform dataframe into sparse matrix
    
    num_users, num_items = data.shape
    item_mean = np.zeros((num_items,1))
    
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = data[:, item_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
            item_mean[item_index] = item_train_mean
            
    df = test[['user_id', 'movie_id']].copy()
    
    df['item_mean_rating'] = np.clip(item_mean[test['movie_id'].values.astype(int)-1], 1, 5)
    
    return df





def computeMFALS(train, test):
    """
    Implementation of matrix factorization using ALS.
    
        train: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'item_mean_rating'.
    
    """
    # parameters
    num_features = 20
    lambda_user = 0.081
    lambda_item = 0.081
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    np.random.seed(1)
    
    data = preprocess_data(train) # transform dataframe into sparse matrix
    
    test_array = test[['user_id', 'movie_id', 'rating']].values.astype(int)

    # initialisation
    user_features, item_features = init_MF(data, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = data.getnnz(axis=0), data.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(data)
    
    while change > stop_criterion:
        user_features = update_user_feature(data, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(data, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        error = compute_error(data, user_features, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

            
    prediction = user_features.T.dot(item_features)
    
    df = test[['user_id', 'movie_id']].copy()
    n = len(df)
    submission = np.zeros((n,1))
    for i in range(n):
        submission[i] = np.clip(prediction[test_array[i][1]-1][test_array[i][0]-1], 1, 5)
    
    df['MF_ALS_rating'] = submission
    
    return df


