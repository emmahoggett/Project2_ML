import numpy as np
import pandas as pd
import tensorflow as tf


from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Add, Activation, Lambda
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
usualCallback = EarlyStopping()

def setDataSet(X, y, ratio):
    """Set global data set and split the data"""
    global X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=ratio, random_state=42)


def optimizerActivationModel(model, layer, neuron, dropout):
    array_model = ['relu', 'sigmoid']
    #array_model = ['relu', 'sigmoid', 'tanh', 'softplus', 'softsign', 'selu']
    activation_mse = []
    for nb_activation in array_model:
        #test activation model to the model and take the optimize number
        model_temp = model
        if (layer == 0):
            model_temp.add(Dense(neuron, input_dim=2, activation=nb_activation))
            model_temp.add(Dropout(dropout))
        else:
            model_temp.add(Dense(neuron, activation=nb_activation))
            model_temp.add(Dropout(dropout))

        model_temp.add(Dense(1, activation='sigmoid')) # Output layer
        model_temp.add(Lambda(lambda x: x * 5))
        
        model_temp.compile(loss='mean_squared_error', optimizer='adam')
        
        history = model_temp.fit(x=X_train, y=y_train,  batch_size=10000, epochs=10000,verbose=0,validation_data=(X_test, y_test),callbacks=[usualCallback])
        
        activation_mse.append(history.history['val_loss'][-1])
    
    
    return array_model[np.argmin(activation_mse)]

def optimizerDropout(model, layer, neuron):
    dropout = np.linspace(0.05,0.95,19)
    dropout_mse = []
    for nb_dropout in dropout:
        #add dropout to the model and take the optimize number
        model_temp = model
        activation_opt = optimizerActivationModel(model, layer, neuron, nb_dropout)
        if (layer == 0):
            model_temp.add(Dense(neuron, input_dim=2, activation=activation_opt))
            model_temp.add(Dropout(nb_dropout))
        else:
            model_temp.add(Dense(neuron, activation=activation_opt))
            model_temp.add(Dropout(nb_dropout))

            
        model_temp.add(Dense(1, activation='sigmoid')) # Output layer
        model_temp.add(Lambda(lambda x: x * 5))
        
        model_temp.compile(loss='mean_squared_error', optimizer='adam')
        
        history = model_temp.fit(x=X_train, y=y_train,  batch_size=10000,epochs=10000,verbose=0,validation_data=(X_test, y_test),callbacks=[usualCallback])
        
        dropout_mse.append(history.history['val_loss'][-1])
    
    return dropout[np.argmin(dropout_mse)], activation_opt


def optimizerNeurons(model, layer):
    nb_max_neurons= 7
    neurons = np.linspace(1, nb_max_neurons*10, num=nb_max_neurons-2)
    neurons = neurons.astype(int)
    neurons_mse = []
    for nb_neuron in neurons:
        #add neurons to the model and take the optimize number
        model_temp = model
        dropout_opt, activation_opt = optimizerDropout(model, layer, nb_neuron)
        if (layer == 0):
            model_temp.add(Dense(nb_neuron, input_dim=2, activation=activation_opt))
            model_temp.add(Dropout(dropout_opt))
        else:
            model_temp.add(Dense(neuron, activation=activation_opt))
            model_temp.add(Dropout(dropout_opt))
        
        model_temp.add(Dense(1, activation='sigmoid')) # Output layer
        model_temp.add(Lambda(lambda x: x * 5))
        
        model_temp.compile(loss='mean_squared_error', optimizer='adam')
        
        history = model_temp.fit(x=X_train, y=y_train,  batch_size=10000,epochs=10000,verbose=0,validation_data=(X_test, y_test),callbacks=[usualCallback])
        
        neurons_mse.append(history.history['val_loss'][-1])
    
    return neurons[np.argmin(neurons_mse)], dropout_opt, activation_opt

def optimizerLayersNeuronsDropoutsModels(X, y):
    """Optimize the number of layers, the number of neurons per layer dropouts and models.
    Test done on:
        layers : from 0 to 20 layers
        neurons: from 1 to 70 with a step of 10
        dropouts: from 0.05 to 0.95 with a step of 0.05
        model: linear regression ('relu') and logistic function ('sigmoid')
    Use 'optimizerNeurons(model, layer)'"""
    layers= range(0,20)
    model = Sequential()
    usualCallback = EarlyStopping()
    current_mse = 1000
    
    setDataSet(X, y, 0.1)

    for nb_layer in layers:
        layers_mse = []
        #add neurons to the model and take the optimize number
        model_temp = model
        neurons_opt, dropout_opt, activation_opt = optimizerNeurons(model, nb_layer)
        if (layer == 0):
            model_temp.add(Dense(neurons_opt, input_dim=2, activation=activation_opt))
            model_temp.add(Dropout(dropout_opt))
        else:
            model_temp.add(Dense(neurons_opt, activation=activation_opt))
            model_temp.add(Dropout(dropout_opt))
        
        model_temp.add(Dense(1, activation='sigmoid')) # Output layer
        model_temp.add(Lambda(lambda x: x * 5))
        
        model_temp.compile(loss='mean_squared_error', optimizer='adam')
        history = model_temp.fit(x=X_train, y=y_train,  batch_size=10000,epochs=10000,verbose=1 ,validation_data=(X_test, y_test),callbacks=[usualCallback])
        
        layers_mse.append(history.history['val_loss'][-1])
  
        if min(layers_mse) < current_mse:
            current_mse = min(layers_mse)
            if nb_layer == 0:
                
                model.add(Dense(neurons_opt, input_dim=2, activation=activation_opt))
                model.add(Dropout(dropout_opt))
            else:
                model.add(Dense(neurons_opt, activation=activation_opt))
                model.add(Dropout(dropout_opt))
        else:
            model.add(Dense(1, activation='sigmoid')) # Output layer
            model.add(Lambda(lambda x: x * 5))
        
            model.compile(loss='mean_squared_error', optimizer='adam')
            # save the model
            model.save('model_optimizerLayerNeuronDropoutModel.h5')
            return model