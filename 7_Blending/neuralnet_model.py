from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras import models
from keras import optimizers
from keras_radam import RAdam
from keras.regularizers import l2

from helpers import*

import pandas as pd
import numpy as np

def computeNeuralNet(train, test, df):
#   Implemenetation of a shallow neural network 

    train_x = 
    
    categorical_train_y = np.zeros([train.shape[0], 5])
    categorical_train_y[np.arange(train.shape[0]), train.rating - 1] = 1
    categorical_train_y.shape

    categorical_test_y = np.zeros([test.shape[0], 5])
    categorical_test_y[np.arange(test.shape[0]), test.rating - 1] = 1
    categorical_test_y.shape
    
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
    
    model.summary()
    
    matrix_fact = model.fit([train.user_id, train.movie_id], train.rating, epochs=20, verbose=1)
    
    prediction = model.predict([test.user_id, test.movie_id])
    
    df['matrix_fact_rating'] = prediction
    
    return df