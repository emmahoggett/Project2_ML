from keras.layers import Input, Dense, Dropout, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Adam

def generate(n_layers, n_neurons, dropout, embedded_users, embedded_movies):
    
    # Input layer
    user = Input(shape=(1,))
    movie = Input(shape=(1,))
    
    # Embedding layers
    u = Embedding(10000, embedded_users, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))
    m = Embedding(1000, embedded_movies, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))
    x = Concatenate()([u, m])
    x = Dropout(dropout)(x)
    
    # Hidden layers
    for i in range(n_layers):
        x = Dense(n_neurons, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    x_ = Dense(5, activation='softmax', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=visible, outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    
    return model