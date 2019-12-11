from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
from keras_radam import RAdam

from helpers import*

import pandas as pd
import numpy as np

def computeMatrixFact(train, test, df):
    """
    Compute Matrix Factorization with keras library
    
        trainset: trainset build to fit the model with the training set.
        test: data frame of the test data.
        df : data frame that will be returned with the prediction in the columns matrix_fact_rating.
    """
    
    print("Start computing Matrix factorization...")
    n_users = len(train.user_id.unique())
    n_movies = len(train.movie_id.unique())
    #creating movie embedding path
    movie_input = Input(shape=[1], name="Movie-Input")
    movie_embedding = Embedding(n_movies+1, 5, name="Movie-Embedding")(movie_input)
    movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

    # creating user embedding path
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    # performing dot product and creating model
    prod = Dot(name="Dot-Product", axes=1)([movie_vec, user_vec])
    model = Model([user_input, movie_input], prod)
    model.compile(RAdam(), 'mean_squared_error')
    
    matrix_fact = model.fit([train.user_id, train.movie_id], train.rating, epochs=20, verbose=1)
    
    prediction = model.predict([test.user_id, test.movie_id])
    
    df['matrix_fact_rating'] = prediction
    print ("... finished")
    
    return df