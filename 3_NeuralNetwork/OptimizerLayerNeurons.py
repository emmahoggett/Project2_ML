import numpy as np
import pandas as pd
import tensorflow as tf


from keras.models import Model,load_model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Concatenate, Dense, Dropout, Add, Activation, Lambda
from keras.callbacks import EarlyStopping

usualCallback = EarlyStopping()

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x


def setDataSet(X, y, ratio):
    """Set global data set and split the data"""
    global X_train_array, X_test_array, y_train, y_test
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=ratio, random_state=42)
    X_train_array = [X_train[:,0], X_train[:,1]]
    X_test_array = [X_test[:,0], X_test[:,1]]
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder_train = encoder.transform(y_train)
    encoder_test = encoder.transform(y_test)

    y_train = np_utils.to_categorical(encoder_train)
    y_test = np_utils.to_categorical(encoder_test)
    
    
def optimizerLayerNeuron(nb_max_layers,nb_max_neurons,n_users, n_movies, n_factors, X, y):
    """Optimize the number of layers and the number of neurons per layer with Rectified 
    Linear Unit Activation Function
    - nb_max_layers: maximum number of layers tested on the model
    - nb_max_neurons: maximum number of neurons multiplied by 10 with a step of 10 tested on one layer on the model"""
    setDataSet(X, y, 0.1)
    
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies, n_factors)(movie)

    break_indicator = 0
    layer = 0
    current_mse=10000
    neurons_array = np.linspace(10, nb_max_neurons*10, num=nb_max_neurons-2)
 
    while break_indicator == 0 and layer < nb_max_layers:
    #loop that add layers to the model until the mse is minimal
        new_layer_mse = []
        for nb_neurons in neurons_array.astype(int):
        #add neurons to the model and take the optimize number
            
            if (layer == 0):
                x_temp = Concatenate()([u, m])
                x_temp = Dropout(0.05)(x_temp)
            else:
                x_temp = Dense(nb_neurons, kernel_initializer='he_normal')(x_temp)
                x_temp = Activation('relu')(x_temp)
                x_temp = Dropout(0.5)(x_temp)
        
            x_temp = Dense(5, kernel_initializer='he_normal')(x_temp) # Output layer
            x_temp = Activation('softmax')(x_temp)
            model_temp = Model(inputs=[user, movie], outputs=x_temp)
            opt = Adam(lr=0.001)
            model_temp.compile(loss='mean_squared_error', optimizer=opt)
        
            history = model_temp.fit(x=X_train_array, y=y_train,  batch_size=1024, 
                             epochs=10000,verbose=1,validation_data=(X_test_array, y_test),callbacks=[usualCallback])
            new_layer_mse.append(history.history['val_loss'][-1])
   
        
        if min(new_layer_mse) < current_mse:
    
            current_mse = min(new_layer_mse)
            print(min(new_layer_mse))
            nb_neurons_opt = (np.argmin(new_layer_mse) + 1)*10

            if layer == 0:
                x = Concatenate()([u, m])
                x = Dropout(0.05)(x)
            else:
                x = Dense(nb_neurons_opt, kernel_initializer='he_normal')(x)
                x = Activation('relu')(x)
                x = Dropout(0.5)(x)
                
            layer += 1
        

        else:
            break_indicator = 1
            x = Dense(5, kernel_initializer='he_normal')(x)
            x = Activation('softmax')(x)
            model = Model(inputs=[user, movie], outputs=x)
            opt = Adam(lr=0.001)
            model.compile(loss='mean_squared_error', optimizer=opt)
            print('Simulation finished successfully')
            return model
            