# -*- coding: utf-8 -*-
"""some functions for computing matrix factorization using stochastic gradient descent."""

import numpy as np
import scipy.sparse as sp

from helpersbaseline import*

def matrix_factorization_SGD(train, num_features = 20, lambda_user=0.07, lambda_item=0.07):
    """matrix factorization by SGD."""
    
    # define parameters
    gamma = 0.02
    num_epochs = 20     # number of full passes through the train set
    
    # set seed
    np.random.seed(1)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col)) # list of the indices of the non zero terms in train

    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

    return user_features, item_features
    
