from matrixfact_model import*
from surprise_model import*
from helpers import*
from neuralnet_model import*

import pandas as pd
import numpy as np

def computeAll (train, valid, test):
    """
    Compute the predictions of Surprise algorithms, the shallow Neural Network and the matrix factorization.
    
        train : data frame of the validation data. Training of the different models is made with this data set.
        valid : data frame of the validation data. The linear regression is made on this data set.
        test: data frame of the test data. Predictions are made on this model.
        
        Return a data frame with the predictions of the test set and print the result of the linear regression.
    """
    
    df_test = computeSurprise(train, test)
    df_test = computeMatrixFact(train, test, df_test)
    df_test = computeNeuralNet(train, test, df_test)
    
    df_valid = computeSurprise(train, valid)
    df_valid = computeMatrixFact(train, valid, df_valid)
    df_valid = computeNeuralNet(train, valid, df_valid)
    models_names = ['knnbasic_user_rating','knnbasic_item_rating','knnmeans_item_rating',
                    'knnmeans_user_rating','slopeone_rating','cocluster_rating', 'matrix_fact_rating', 'svdpp_rating',
                   'svd_rating','nmf_rating', 'neural_net_rating']
    
    def blending_funct(x):

        df_valid['weighted'] = df_valid[models_names[0]]*x[0]

        for i in range(1,len(models_names)):
            df_valid['weighted'] = df_valid[models_names[i]] *  x[i] + df_valid['weighted']
    
        rmse =  ((df_valid.weighted -valid.rating) ** 2).mean() ** .5
        return rmse

    def blending():
        x0 = 1/(len(models_names))*np.ones(len(models_names))

        res = sc.optimize.minimize(blending_funct, x0, method='SLSQP', options={'disp':True})
        print(res)
    
    blending()
    
    return df_test
    
    