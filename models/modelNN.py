from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras import models
from keras import optimizers
from keras_radam import RAdam
from keras.regularizers import l2

from helpers import*

import pandas as pd
import numpy as np


############################################################################
#
#   In this file, the following method are implemented to increase readability
#   of the code:
#      - Deep Neural Network
#      - Shallow Neural Network
#   The two methods are implemented in the function computeNN
#
###########################################################################


def computeNN(train, test):
    """ Implementation of a deep and a shallow neural network.
    
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the columns 'NN_shallow_rating' 
        and 'NN_deep_rating'.
    
    """
    print ("Start to compute a shallow neural network...")
    shallow_NN = computeShallowNN(train, test)
    print ("... Finished sucessfully")
    
    print ("Start to compute a deep neural network...")
    deep_NN = computeDeepNN(train, test)
    print ("... Finished sucessfully")
    
    NN_rating = shallow_NN\
                .merge(deep_NN, on=['user_id', 'movie_id'])
    
    return NN_rating

def computeShallowNN(train, test):
    """
    Implementation of a shallow neural network.
    
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'NN_shallow_rating'.
    
    """
    n_items = 1000
    n_users = 10000
    
    categorical_train_y = np.zeros([train.shape[0], 5])
    categorical_train_y[np.arange(train.shape[0]), train.rating - 1] = 1
    

    categorical_test_y = np.zeros([test.shape[0], 5])
    categorical_test_y[np.arange(test.shape[0]), test.rating - 1] = 1
    
    
    features = 48
    
    input_i = layers.Input(shape=[1])
    i = layers.Embedding(n_items + 1, features)(input_i)
    i = layers.Flatten()(i)
    i = layers.normalization.BatchNormalization()(i)

    input_u = layers.Input(shape=[1])
    u = layers.Embedding(n_users + 1, features)(input_u)
    u = layers.Flatten()(u)
    u = layers.normalization.BatchNormalization()(u)
    
    nn = layers.concatenate([i, u])
    
    nn = layers.Dense(512, activation='relu')(nn)
    nn = layers.Dropout(0.5)(nn)
    nn = layers.normalization.BatchNormalization()(nn)
    
    nn = layers.Dense(128, activation='relu')(nn)
    
    output = layers.Dense(5, activation='softmax')(nn)
    
    model = models.Model([input_i, input_u], output)
    model.compile(optimizer='adamax', loss='categorical_crossentropy')
    
    
    history = model.fit([train.user_id, train.movie_id], y=categorical_train_y,  batch_size=20480, epochs=20)
    
    prediction = model.predict([test.user_id, test.movie_id])
    df = test[['user_id', 'movie_id']].copy()
    df['NN_shallow_rating'] = np.dot(prediction,[1,2, 3, 4, 5])
    
    return df


def computeDeepNN(train, test):
    """
    Implementation of a deep neural network.
    
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        
        return : data frame that will be returned with the prediction in the column 'NN_deep_rating'.
    
    """
    n_items = 1000
    n_users = 10000
    
    categorical_train_y = np.zeros([train.shape[0], 5])
    categorical_train_y[np.arange(train.shape[0]), train.rating - 1] = 1
    

    categorical_test_y = np.zeros([test.shape[0], 5])
    categorical_test_y[np.arange(test.shape[0]), test.rating - 1] = 1
    
    features = 48
    
    input_i = layers.Input(shape=[1])
    i = layers.Embedding(n_items + 1, features)(input_i)
    i = layers.Flatten()(i)
    i = layers.normalization.BatchNormalization()(i)

    input_u = layers.Input(shape=[1])
    u = layers.Embedding(n_users + 1, features)(input_u)
    u = layers.Flatten()(u)
    u = layers.normalization.BatchNormalization()(u)
    
    nn = layers.concatenate([i, u])
    
    nn = layers.Dense(1024, activation='relu')(nn)
    nn = layers.Dropout(0.5)(nn)
    nn = layers.normalization.BatchNormalization()(nn)
    nn = layers.Dense(512, activation='relu')(nn)
    nn = layers.Dropout(0.5)(nn)
    nn = layers.normalization.BatchNormalization()(nn)
    nn = layers.Dense(256, activation='relu')(nn)
    nn = layers.Dropout(0.5)(nn)
    nn = layers.normalization.BatchNormalization()(nn)
    nn = layers.Dense(128, activation='relu')(nn)
    
    output = layers.Dense(5, activation='softmax')(nn)
    
    model = models.Model([input_i, input_u], output)
    model.compile(optimizer='adamax', loss='categorical_crossentropy')
    
    history = model.fit([train.user_id, train.movie_id], y=categorical_train_y,  batch_size=20480, epochs=20)
    
    prediction = model.predict([test.user_id, test.movie_id])
    
    df = test[['user_id', 'movie_id']].copy()
    df['NN_deep_rating'] = np.dot(prediction,[1,2, 3, 4, 5])
    return df