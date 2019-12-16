import pandas as pd
import numpy as np

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import Dataset, Reader

############################################################################
#
#   In this file, the following method are implemented to increase readability
#   of the code:
#      -k-NN with means
#      -k-NN basic
#      -k-NN with Z score
# k-NN correspond to the k Nearest Neighbors method
#
##########################################################################

def dataTrainSurprise(data_np, test_np):
    """Prepare the data set for training on the surprise library """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_np[['user_id', 'movie_id', 'rating']], reader=reader)
    trainset = data.build_full_trainset()
    
    test = test_np[['user_id', 'movie_id']].copy()
    return trainset, test


def computeKNNBasicUser(data, test_np, test_purpose = False):
    """Compute the k-NN basic user based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : MSD, user based
         - Number of closest neighbors : 253"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': True}
    knnbasic_algo = KNNBasic(k = 253, sim_options =sim_options).fit(trainset)

    test['knnbasic_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    if test_purpose:
        test.to_csv('knnbasic_user_test.csv')
    else:
        test.to_csv('knnbasic_user_submission.csv')

def computeKNNBasicMovie(data, test_np, test_purpose = False):
    """Compute the k-NN basic item based method and return the predictions on the test into a file
     The method is on all the data and got the following settings:
         - Similarity function : MSD, item based
         - Number of closest neighbors : 23"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': False}
    knnbasic_algo = KNNBasic(k = 23, sim_options =sim_options).fit(trainset)
    
    test['knnbasic_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    if test_purpose:
        test.to_csv('knnbasic_item_test.csv')
    else:
        test.to_csv('knnbasic_item_submission.csv')
    

def computeKNNMeansUser(data, test_np, test_purpose = False):
    """Compute the k-NN with mean user based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : MSD, user based
         - Number of closest neighbors : 500"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': True}
    knnmeans_algo = KNNWithMeans(k = 500, sim_options =sim_options).fit(trainset)

    test['knnmeans_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    if test_purpose:
        test.to_csv('knnmeans_user_test.csv')
    else:
        test.to_csv('knnmeans_user_submission.csv')
    
def computeKNNMeansMovie(data, test_np, test_purpose = False):
    """Compute the k-NN with mean item based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : Pearson Baseline, item based
         - Number of closest neighbors : 108"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'pearson_baseline','user_based': False}
    knnmeans_algo = KNNWithMeans(k = 108, sim_options =sim_options).fit(trainset)

    test['knnmeans_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    if test_purpose:
        test.to_csv('knnmeans_item_test.csv')
    else:
        test.to_csv('knnmeans_item_submission.csv')

def computeKNNZScoreUser(data, test_np, test_purpose = False):
    """Compute the k-NN with z score user based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : MSD, user based
         - Number of closest neighbors : 500"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': True}
    knnz_algo = KNNWithZScore(k = 500, sim_options =sim_options).fit(trainset)

    test['knnzscore_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnz_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    if test_purpose:
        test.to_csv('knnzscore_user_test.csv')
    else:
        test.to_csv('knnzscore_user_submission.csv')

def computeKNNZScoreMovie(data, test_np, test_purpose = False):
    """Compute the k-NN with z score item based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : Pearson baseline, item based
         - Number of closest neighbors : 108"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'pearson_baseline','user_based': False}
    knnz_algo = KNNWithZScore(k = 108, sim_options =sim_options).fit(trainset)

    test['knnzscore_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnz_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    if test_purpose:
        test.to_csv('knnzscore_item_test.csv')
    else:
        test.to_csv('knnzscore_item_submission.csv')
    
