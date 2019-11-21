from keras.layers import Input, Dense, Dropout, Embedding, Concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from helpersNeuronalNet import*

def generate(n_layers, n_neurons, dropout, n_users, n_movies, n_factors):
    
    # Input layer
    user = Input(shape=(1,))
    movie = Input(shape=(1,))
    
    # Embedding layers
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    m = EmbeddingLayer(n_movies, n_factors)(movie)

    x = Concatenate()([u, m])
    x = Dropout(dropout)(x)
    
    # Hidden layers
    for i in range(n_layers):
        x = Dense(n_neurons, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    x = Dense(5, activation='softmax', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model