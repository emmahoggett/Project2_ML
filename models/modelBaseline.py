import tensorflow as tf
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import pandas as pd
from itertools import groupby


def computeGlobalMean(data, test):
    df = test[['user_id', 'item_id']].copy()
    
    global_mean = data[['rating']].mean()
    df['global_mean_rating']=df[['user_id', 'movie_id']] \
    .apply(lambda row: global_mean, axis=1)
    
    return df