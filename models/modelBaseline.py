import tensorflow as tf
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import pandas as pd
from itertools import groupby

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
    
    print ("Starting to compute MF using SGD...")
    MF_SGD = computeMFSGD(train, test)
    print ("... Finished sucessfully")

    print ("Starting to compute MF using ALS...")
    MF_ALS = computeMFALS(train, test)
    print ("... Finished sucessfully")    
    
    mean_rating = global_mean\
                .merge(user_mean, on=['user_id', 'movie_id'])\
                .merge(item_mean, on=['user_id', 'movie_id'])\
                .merge(MF_SGD, on=['user_id', 'movie_id'])\
                .merge(MF_ALS, on=['user_id', 'movie_id'])
    
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
    num_users, num_items = train.shape
    user_mean = np.zeros((num_users,1))

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[user_index, :]
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
    num_users, num_items = train.shape
    item_mean = np.zeros((num_items,1))
    
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[:, item_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
            item_mean[item_index] = item_train_mean
            
    df = test[['user_id', 'movie_id']].copy()
    
    df['item_mean_rating'] = np.clip(item_mean[test['movie_id'].values.astype(int)-1], 1, 5)
    
    return df


###################################################


def computeMFSGD(train, test):
    """
    Implementation of matrix factorization using SGD.
    
        train: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'item_mean_rating'.
    
    """
    # define parameters
    num_features = 20
    lambda_user = 0.07
    lambda_item = 0.07
    gamma = 0.02
    num_epochs = 20     # number of full passes through the train set
    np.random.seed(1)     # set seed

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col)) # list of the indices of the non zero terms in train

    for it in range(num_epochs):        
        np.random.shuffle(nz_train)
        
        gamma /= 1.2  # decrease step size
        
        for d, n in nz_train:
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

    prediction = user_features.T.dot(item_features)
    
    df = test[['user_id', 'movie_id']].copy()
    n = len(df)
    submission = np.zeros((n,1))
    for i in range(n):
        submission[i] = np.clip(prediction[test[i][1]-1][test[i][0]-1], 1, 5)
    
    df['MF_SGD_rating'] = submission
    
    return df



def computeMFALS(train, test):
    """
    Implementation of matrix factorization using ALS.
    
        train: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'item_mean_rating'.
    
    """
    # parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    np.random.seed(1)

    # initialisation
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    
    while change > stop_criterion:
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)

            
    prediction = user_features.T.dot(item_features)
    
    df = test[['user_id', 'movie_id']].copy()
    n = len(df)
    submission = np.zeros((n,1))
    for i in range(n):
        submission[i] = np.clip(prediction[test[i][1]-1][test[i][0]-1], 1, 5)
    
    df['MF_ALS_rating'] = submission
    
    return df

###################################################
### Functions needed for Matrix factorisation #####


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
        
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features

def update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
        
    return updated_user_features


def update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
        
    return updated_item_features


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value])) for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value])) for g, value in grouped_nz_train_bycol]
    
    return nz_train, nz_row_colindices, nz_col_rowindices


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data
