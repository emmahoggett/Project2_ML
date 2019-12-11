from surprise import KNNBasic, KNNWithMeans
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering
from surprise import Dataset, Reader

from helpers import*

import pandas as pd
import numpy as np


def computeSurprise(train, test):
    """
    Compute the following algorithms of the surprise library:
        - SVD
        - SVD++
        - Non-negative Matric factorization
        - k-Nearest Neighbours basic user based and item based
        - k-Nearest Neighbours with mean user based and item based
        - Slope one
        - Co-Clustering
    All those model are optimized.
    
        train : pandas data frame, that define the training set. The training set is used to fit models.
        test : pandas data frame, that define the test set. Predictions will be made on the data set.
    
    """
    reader = Reader(rating_scale=(1,5))
    df = test[['user_id', 'movie_id']].copy()
    
    data = Dataset.load_from_df(train[['user_id', 'movie_id', 'rating']], reader)
    
    trainset = data.build_full_trainset()
    
    df = computeKNNBasic(trainset, test, df)
    df = computeSVD(trainset, test, df)
    df = computeSVDpp(trainset, test, df)
    df = computeNMF(trainset, test, df)
    df = computeKNNMeans(trainset, test, df)
    df = computeSlopeOne (trainset, test, df)
    df = computeCoClustering(trainset, test, df)
    
    return df

def computeSVD(trainset, test, df):
    """ 
    Compute the SVD algorithm in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the column svd_rating.
    """
    print("Start computing SVD...")
    svd = SVD().fit(trainset)
    df['svd_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: svd.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    return df

def computeSVDpp(trainset, test, df):
    """ 
    Compute the SVD++ algorithm in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the column svdpp_rating.
    """
    
    print("Start computing SVDpp...")
    svdpp = SVDpp().fit(trainset)
    df['svdpp_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: svdpp.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    return df

def computeNMF(trainset, test, df):
    """ 
    Compute the Non-negative Matrix factorization in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the column nmf_rating.
    """
    
    print("Start computing NMF...")
    nmf = NMF().fit(trainset)
    df['nmf_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: nmf.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    return df
    
def computeKNNBasic(trainset, test, df):
    """ 
    Compute the item based and the user based Basic k-Nearest Neighbours in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the columns knnbasic_user_rating and
        knnbasic_item_rating.
        
     """
    
    print("Start computing KNNBasic...")
    simoption = {'user_based': True}
    knnbasic_user = KNNBasic(k = 253, sim_options=simoption).fit(trainset)

    df['knnbasic_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_user.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    simoption = {'user_based': False}
    knnbasic_item = KNNBasic(k = 23, sim_options=simoption).fit(trainset)

    df['knnbasic_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_item.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    return df


def computeKNNMeans(trainset, test, df):
    """ 
    Compute the item based and the user based k-Nearest Neighbours with mean in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the columns knnmeans_user_rating and
        knnmeans_item_rating.
        
     """
    print("Start computing KNNMeans...")
    sim_options = {'name': 'pearson_baseline', 'user_based': True}
    knnmeans_user = KNNWithMeans(k = 500, sim_options = sim_options).fit(trainset)
    
    df['knnmeans_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_user.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    knnmeans_item = KNNWithMeans(k = 108, sim_options = sim_options).fit(trainset)
    
    df['knnmeans_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_item.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    
    return df

def computeSlopeOne (trainset, test, df):
    """ 
    Compute the Slope One algorithm in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the columns slopeone_rating.
        
     """
    print("Start computing SlopeOne...")
    slopeone = SlopeOne().fit(trainset)
    
    df['slopeone_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: slopeone.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    return df

def computeCoClustering(trainset, test, df):
    """ 
    Compute the Co-Clustering algorithm in Surprise library
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the columns cocluster_rating.
        
     """
    print("Start computing CoClustering...")
    cocluster = CoClustering(n_cltr_u=2, n_cltr_i=19, n_epochs = 30).fit(trainset)

    df['cocluster_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: cocluster.predict(row['user_id'], row['movie_id'])[3], axis=1)
    print ("... finished")
    
    return df