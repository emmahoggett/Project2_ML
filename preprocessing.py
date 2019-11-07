import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

#For reproductibility, remove seed from split_data

def split_data(ratings,min_num_ratings,p_test=0.1):
    """The function return the users that give more then 10 advices
    ratings: first column represent the user and the film
          second column represent the grading
    min_num_ratings: minimum number of ratings given by a user
    p_test: proportion of the data that will be use for test"""
    # determine the number of items per user and the # of users per item
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    
    # set seed 
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users] 
    
    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    nz_items, nz_users = valid_ratings.nonzero()
    
    
    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]
        
    return valid_ratings, train, test