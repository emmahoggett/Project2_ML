import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model, Sequential, load_model

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras_radam import RAdam

def computeMF(train, test):
    """ Implementation of a matrix factorisation.
    
        train: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the columns 'MF_RMSE_rating'.
    
    """
    
    print ("Starting to compute global mean...")
    MF = computeMF(train, test)
    print ("... Finished sucessfully")
    
    return MF

###################################################

def computeMF(train, test):
    """
    Implementation of matrix factorisation using keras library.
    
        train: train set build to fit the model.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'MF_RMSE_rating'.
    
    """
    K = 25

    #creating movie embedding path
    movie_input = Input(shape=[1], name="Movie-Input")
    movie_embedding = Embedding(n_movies+1, K, name="Movie-Embedding", embeddings_regularizer=l2(1e-6))(movie_input)
    movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

    # creating user embedding path
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, K, name="User-Embedding", embeddings_regularizer=l2(1e-6))(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    # performing dot product and creating model
    prod = Dot(name="Dot-Product", axes=1)([movie_vec, user_vec])
    model = Model([user_input, movie_input], prod)
    model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mse', 'accuracy'])
    
    history = model.fit([train.user_id, train.movie_id], train.rating, batch_size=1000, epochs=20, verbose=1)
    
    prediction = model.predict([test.user_id, test.movie_id])
    
    df = test[['user_id', 'movie_id']].copy()
    df['MF_RMSE_rating'] = prediction
    
    return df

