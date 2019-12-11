from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering
from surprise import Dataset, Reader

from helpers import*

import pandas as pd
import numpy as np


def computeSurprise(train, test):
    reader = Reader(rating_scale=(1,5))
    df = test[['user_id', 'movie_id']].copy()
    
    data = Dataset.load_from_df(train[['user_id', 'movie_id', 'rating']], reader)
    
    trainset = data.build_full_trainset()
    
    df = computeKNNBasic(trainset, test, df)
    df = computeKNNMeans(trainset, test, df)
    df = computeSlopeOne (trainset, test, df)
    df = computeCoClustering(trainset, test, df)
    
    return df
    
def computeKNNBasic(trainset, test, df):
    simoption = {'user_based': True}
    knnbasic_user = KNNBasic(k = 253, sim_options=simoption).fit(trainset)

    df['knnbasic_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_user.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    simoption = {'user_based': False}
    knnbasic_item = KNNBasic(k = 23, sim_options=simoption).fit(trainset)

    df['knnbasic_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnbasic_item.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return df

def computeKNNMeans(trainset, test, df):
    sim_options = {'name': 'pearson_baseline', 'user_based': True}
    knnmeans_user = KNNWithMeans(k = 500, sim_options = sim_options).fit(trainset)
    
    df['knnmeans_user_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_user.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    knnmeans_item = KNNWithMeans(k = 108, sim_options = sim_options).fit(trainset)
    
    df['knnmeans_item_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: knnmeans_item.predict(row['user_id'], row['movie_id'])[3], axis=1)
    return df

def computeSlopeOne (trainset, test, df):
    slopeone = SlopeOne().fit(trainset)
    
    df['slopeone_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: slopeone.predict(row['user_id'], row['movie_id'])[3], axis=1)
    return df

def computeCoClustering(trainset, test, df):
    cocluster = CoClustering(n_cltr_u=2, n_cltr_i=19, n_epochs = 30).fit(trainset)

    df['cocluster_rating'] = test[['user_id', 'movie_id']] \
    .apply(lambda row: cocluster.predict(row['user_id'], row['movie_id'])[3], axis=1)
    
    return df