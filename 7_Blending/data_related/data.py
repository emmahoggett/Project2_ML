from itertools import groupby
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_TRAIN_PATH = 'data_train.csv'
SUBMISSION_PATH = 'sample_submission.csv'
TEST_SET_SIZE = 0.2

class Data():
    """
    Class represent the dataset by loading the csv files into various
    data structures used throughout the project.
    """

    def __init__(self, data_train_path=None, data_test_path=None, test_purpose=True, blend_purpose = True):
        """
        Initializes the internal data structures and statistics
        Args:
            data_train_path: The specified path for the csv file
                containing the training dataset
            data_test_path: The specified path for the csv file
                containing the test dataset
            test_purpose: True for testing, False for creating submission
        """
        print('Preparing data ...')
        if data_train_path is None:
            data_train_path = DATA_TRAIN_PATH
        if data_test_path is None:
            data_test_path = SUBMISSION_PATH
        
        if test_purpose:
            print('Splitting data to train and test data ...')
            data_df = pd.read_csv(data_train_path)
            split = data_df.Id.astype(str).str.split(r"r|_c", expand=True)
            prediction = data_df[['Prediction']]
            data_df.drop(['Id','Prediction'],axis = 1, inplace = True)
            data_df['user_id'] = split[1]
            data_df['movie_id'] = split[2]
            data_df['rating'] = prediction
            
            self.train_df, test_df = train_test_split(data_df, test_size=TEST_SET_SIZE, random_state=1)
            
            if blend_purpose:
                self.valid_df, self.test_df = train_test_split(test_df, test_size=0.5, random_state=1)
            else:
                self.test_df = test_df
            
            print('... data is splitted.')
        else:
            data_df = pd.read_csv(data_train_path)
            split = data_df.Id.astype(str).str.split(r"r|_c", expand=True)
            prediction = data_df[['Prediction']]
            data_df.drop(['Id','Prediction'],axis = 1, inplace = True)
            data_df['user_id'] = split[1]
            data_df['movie_id'] = split[2]
            data_df['rating'] = prediction
            
            self.data_df = data_df
            
            test_df = pd.read_csv(data_test_path)
            split = test_df.Id.astype(str).str.split(r"r|_c", expand=True)
            prediction = test_df[['Prediction']]
            test_df.drop(['Id','Prediction'],axis = 1, inplace = True)
            test_df['user_id'] = split[1]
            test_df['movie_id'] = split[2]
            test_df['rating'] = prediction
            
            self.test_df = test_df
            
        print('... data is prepared.')
