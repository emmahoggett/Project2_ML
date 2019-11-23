import numpy as np
import pandas as pd
import tensorflow as tf


from keras.models import Model,load_model
from keras.layers import Input, Dot
from keras.optimizers import Adam
from keras.layers import Concatenate, Dense, Dropout, Add, Activation, Lambda
from keras.callbacks import EarlyStopping

usualCallback = EarlyStopping()


from helpersNeuronalNet import*
from model_generation import*
    
    
def layers(max_nb_layers, nb_neurons, nb_dropout, nb_embeddinglayer, data):
    layer = 0
    max_accuracy = 0
    break_ind = 0
    X_train_array, X_test_array, y_train, y_test, n_movies, n_users = setDataSet(data)
    accuracy = []
    
    for layer in range(max_nb_layers):
        model = generate(layer, nb_neurons, nb_dropout, n_users, n_movies, nb_embeddinglayer)
        
        history = model.fit(x=X_train_array, y=y_train,  batch_size=1024, 
                             epochs=10000,verbose=1,validation_data=(X_test_array, y_test),callbacks=[usualCallback])
        accuracy.append(history.history['val_acc'][-1])
        
        if (max_accuracy < max(accuracy)):
            max_accuracy = max(accuracy)
            nb_layers = layer
        
    return nb_layers, max_accuracy

def neurons(nb_layers, max_nb_neurons, nb_dropout, nb_embeddinglayer, data):
    
    X_train_array, X_test_array, y_train, y_test, n_movies, n_users = setDataSet(data)
    
    nb_neurons = 50
    max_accuracy = 0
    break_ind = 0
    accuracy = []
    
    for neuron in range(50,max_nb_neurons,10):
        model = generate(nb_layers, neuron, nb_dropout, n_users, n_movies, nb_embeddinglayer)
        
        history = model.fit(x=X_train_array, y=y_train,  batch_size=1024, 
                             epochs=10000,verbose=1,validation_data=(X_test_array, y_test),callbacks=[usualCallback])
        accuracy.append(history.history['val_acc'][-1])
        
        if (max_accuracy < max(accuracy)):
            max_accuracy = max(accuracy)
            nb_neurons = neuron
        
        
    return nb_neurons,max_accuracy 

def embeddinglayer(nb_layers, nb_neurons, nb_dropout, max_nb_embeddinglayer, data):
    X_train_array, X_test_array, y_train, y_test, n_movies, n_users = setDataSet(data)
    max_accuracy = 0
    accuracy = []
    for embeddinglayer in range(10,max_nb_embeddinglayer,10):
        model = generate(nb_layers, nb_neurons, nb_dropout, n_users, n_movies, embeddinglayer)
        history = model.fit(x=X_train_array, y=y_train,  batch_size=1024, 
                             epochs=10000,verbose=1,validation_data=(X_test_array, y_test),callbacks=[usualCallback])
        accuracy.append(history.history['val_acc'][-1])
        
        if (max_accuracy < max(accuracy)):
            max_accuracy = max(accuracy)
            nb_embeddinglayer = embeddinglayer
            
    return nb_embeddinglayer, max_accuracy 

def optimize_dropout(n_layers, n_neurons, n_factors, data):
    
    dropouts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    accuracy = []
    max_accuracy = 0
    X_train_array, X_test_array, y_train, y_test, n_movies, n_users = setDataSet(data)
    for dropout in dropouts:
        model = generate(n_layers, n_neurons, dropout, n_users, n_movies, n_factors)
        history = model.fit(x=X_train_array, y=y_train,  batch_size=1024, \
                            epochs=10000,verbose=1, validation_data=(X_test_array, y_test), callbacks=[usualCallback])
        accuracy.append(history.history['val_acc'][-1])
        
        if (max_accuracy < max(accuracy)):
            max_accuracy = max(accuracy)
            dropout_opt = embeddinglayer
    
    return dropout_opt, max_accuracy