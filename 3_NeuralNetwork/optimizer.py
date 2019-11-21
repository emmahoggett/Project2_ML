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
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, nb_embeddinglayer)(user)
    
    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies, nb_embeddinglayer)(movie)

    while ((layer <= max_nb_layers) & (break_ind == 0)):
        accuracy = []
        
        if (layer == 0):
            x = Concatenate()([u, m])
        else:
            x = Dense(nb_neurons, kernel_initializer='he_normal')(x)
            x = Activation('relu')(x)
        
        x = Dense(5, kernel_initializer='he_normal')(x) # Output layer
        x = Activation('softmax')(x)
        model = Model(inputs=[user, movie], outputs=x)
        opt = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        history = model.fit(x=X_train_array, y=y_train,  batch_size=1024, 
                             epochs=10000,verbose=1,validation_data=(X_test_array, y_test),callbacks=[usualCallback])
        accuracy.append(history.history['val_accuracy'][-1])
        layer+=1
        
        if (max_accuracy > max(accuracy)):
            break_ind = 1
        
        else:
            max_accuracy = max(accuracy)
        
    return layer

def neurons(nb_layers, max_nb_neurons, nb_dropout, nb_embeddinglayer, data):
    
    X_train_array, X_test_array, y_train, y_test, n_movies, n_users = setDataSet(data)
    
    nb_neurons = 50
    max_accuracy = 0
    break_ind = 0
    
    for neuron in range(50,max_nb_neurons,10):
        model = generate(nb_layers, neuron, nb_dropout, n_users, n_movies, nb_embeddinglayer)
        
        history = model.fit(x=X_train_array, y=y_train,  batch_size=1024, 
                             epochs=10000,verbose=1,validation_data=(X_test_array, y_test),callbacks=[usualCallback])
        accuracy.append(history.history['val_accuracy'][-1])
        
        if (max_accuracy < max(accuracy)):
            max_accuracy = max(accuracy)
            nb_neurons = neuron
        
        
    return nb_neurons