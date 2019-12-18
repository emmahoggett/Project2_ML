import pandas as pd
import numpy as np


from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SlopeOne
from surprise import SVD, SVDpp
from surprise import CoClustering

from surprise import Dataset, Reader


############################################################################
#
#   In this file, the following method are implemented to increase readability
#   of the code:
#      - k-NN with means
#      - k-NN basic
#      - k-NN with Z score
#      - SVD
#      - SVD++
#      - Slope one
#      - Co-clustering
# k-NN correspond to the k Nearest Neighbors method
#
###########################################################################


def computeSurprise(data, test_np):
    """ Compute the following method :
            - k-NN with means
            - k-NN basic
            - k-NN with Z score
            - SVD
            - SVD++
            - Slope one
            - Co-clustering
        
           data : data frame which represent the train set
           test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of predictions for each model"""
    
    print ("Start to compute k-NN Basic ...")
    knn_basic_user = computeKNNBasicUser(data, test_np)
    knn_basic_item = computeKNNBasicMovie(data, test_np)
    print ("... Finished sucessfully")
    
    
    print ("Start to compute k-NN with Means ...")
    knn_means_user = computeKNNMeansUser(data, test_np)
    knn_means_item = computeKNNMeansMovie(data, test_np)
    print ("... Finished sucessfully")
    
    print ("Start to compute k-NN ZScore ...")
    knn_zscore_user = computeKNNZScoreUser(data, test_np)
    knn_zscore_item = computeKNNZScoreMovie(data, test_np)
    print ("... Finished sucessfully")
    
    print ("Start to compute Slope one ...")
    slopeone = computeSlopeOne(data, test_np)
    print ("... Finished sucessfully")
    
    print ("Start to compute Co-clustering ...")
    coclustering = computeCoClustering(data, test_np)
    print ("... Finished sucessfully")
    
    print ("Start to compute biased SVD ...")
    svd = computeSVDBiased(data, test_np)
    print ("... Finished sucessfully")
    
    print ("Start to compute unbiased SVD ...")
    mf = computeSVDUnbiased(data, test_np)
    print ("... Finished sucessfully")
    
    print ("Start to compute SVD++ ...")
    svdpp = computeSVDpp(data, test_np)
    print ("... Finished sucessfully")
    
    surprise_ratings = knn_basic_user \
                        .merge(knn_basic_item, on=['user_id', 'movie_id'])\
                        .merge(knn_means_user, on=['user_id', 'movie_id'])\
                        .merge(knn_means_item, on=['user_id', 'movie_id'])\
                        .merge(knn_zscore_user, on=['user_id', 'movie_id'])\
                        .merge(knn_zscore_item, on=['user_id', 'movie_id'])\
                        .merge(slopeone, on=['user_id', 'movie_id'])\
                        .merge(coclustering, on=['user_id', 'movie_id'])\
                        .merge(svd, on=['user_id', 'movie_id'])\
                        .merge(mf, on=['user_id', 'movie_id'])\
                        .merge(svdpp, on=['user_id', 'movie_id'])
    
    return surprise_ratings
    

def dataTrainSurprise(data_np, test_np):
    """Prepare the data set for training on the surprise library """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_np[['user_id', 'movie_id', 'rating']], reader=reader)
    trainset = data.build_full_trainset()
    
    test = test_np[['user_id', 'movie_id']].copy()
    return trainset, test


def computeKNNBasicUser(data, test_np):
    """Compute the k-NN basic user based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : MSD, user based
         - Number of closest neighbors : 253
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'knnbasic_user_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': True}
    knnbasic_algo = KNNBasic(k = 253, sim_options =sim_options).fit(trainset)

    test['knnbasic_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeKNNBasicMovie(data, test_np):
    """Compute the k-NN basic item based method and return the predictions on the test into a file
     The method is on all the data and got the following settings:
         - Similarity function : MSD, item based
         - Number of closest neighbors : 23
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'knnbasic_item_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': False}
    knnbasic_algo = KNNBasic(k = 23, sim_options =sim_options).fit(trainset)
    
    test['knnbasic_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test
    

def computeKNNMeansUser(data, test_np):
    """Compute the k-NN with mean user based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : MSD, user based
         - Number of closest neighbors : 500
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'knnmeans_user_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': True}
    knnmeans_algo = KNNWithMeans(k = 500, sim_options =sim_options).fit(trainset)

    test['knnmeans_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test
    
def computeKNNMeansMovie(data, test_np):
    """Compute the k-NN with mean item based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : Pearson Baseline, item based
         - Number of closest neighbors : 108
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'knnmeans_item_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'pearson_baseline','user_based': False}
    knnmeans_algo = KNNWithMeans(k = 108, sim_options =sim_options).fit(trainset)

    test['knnmeans_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeKNNZScoreUser(data, test_np):
    """Compute the k-NN with z score user based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : MSD, user based
         - Number of closest neighbors : 500
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'knnzscore_user_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'msd','user_based': True}
    knnz_algo = KNNWithZScore(k = 500, sim_options =sim_options).fit(trainset)

    test['knnzscore_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnz_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeKNNZScoreMovie(data, test_np):
    """Compute the k-NN with z score item based method and return the predictions on the test
     The method is on all the data and got the following settings:
         - Similarity function : Pearson baseline, item based
         - Number of closest neighbors : 108
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'knnzscore_item_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    sim_options = {'name':'pearson_baseline','user_based': False}
    knnz_algo = KNNWithZScore(k = 108, sim_options =sim_options).fit(trainset)

    test['knnzscore_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnz_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test


def computeSlopeOne(data, test_np):
    """Compute the slope one method and return the predictions on the test
     The method has no parameter.
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'slopeone_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    slopeone = SlopeOne().fit(trainset)
    
    test['slopeone_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: slopeone.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeCoClustering(data, test_np):
    """Compute the co-clustering method and return the predictions on the test
     The method has the following parameter:
         - Number of user clusters : 2
         - Number of item clusters : 19
         - Number of epochs: 30
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'cocluster_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    cocltr_algo = CoClustering(n_cltr_u=2, n_cltr_i=19, n_epochs = 30).fit(trainset)
    
    test['cocluster_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: cocltr_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeSVDBiased(data, test_np):
    """Compute the biased SVD method and return the predictions on the test
     The method has the following parameter:
         - Number of factors : 6
         - All regularization parameter : 0.025
         - With biased terms
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'svd_rating'"""
    
    trainset, test = dataTrainSurprise(data, test_np)
    
    svdbiased_algo = SVD (n_factors = 6, reg_all=0.025).fit(trainset)
    
    test['svd_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: svdbiased_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeSVDUnbiased(data, test_np):
    """Compute the unbiased SVD method and return the predictions on the test
     The method has the following parameter:
         - Number of factors : 6
         - All regularization parameter : 0.025
         - No biased terms
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'mf_rating'"""
    trainset, test = dataTrainSurprise(data, test_np)
    
    mf_algo = SVD(biased=False, n_factors = 6, reg_all=0.025).fit(trainset)
    
    test['mf_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: mf_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test

def computeSVDpp(data, test_np):
    """Compute the SVD++ method and return the predictions on the test
     The method has the following parameter:
         - Number of factors : 6
         - All regularization parameter : 0.025
         
         data : data frame which represent the train set
         test_np : data frame on which the prediction will be returned
         
         return : test_np with a column of prediction named 'svdpp_rating'"""
    trainset, test = dataTrainSurprise(data, test_np)
    
    svdpp_algo = SVDpp(n_factors = 6, reg_all=0.025).fit(trainset)
    
    test['svdpp_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: svdpp_algo.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return test