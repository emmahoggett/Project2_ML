import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import RidgeCV

from models.modelNN import*
from models.modelBaseline import*
from models.modelMatrixFact import*
from models.modelSurprise import*

from sklearn.model_selection import KFold
from helpers import create_csv, load_data


# Load the train set and the submission set
samples = load_data('data/sampleSubmission.csv')
data = load_data('data/data_train.csv')

# Train and make predictions on the surprise, neural network, baseline and matrix factorization method
surprise_ratings = computeSurprise(data, samples)
NN_ratings = computeNN(data, samples)
baseline_ratings = computeBaseline(data, samples)
MF_ratings = computeMF(data, samples)

# Merge the predictions
submission_pred = surprise_ratings \
                    .merge(NN_ratings, on=['user_id', 'movie_id'])\
                    .merge(baseline_ratings, on = ['user_id', 'movie_id'])\
                    .merge(MF_ratings, on = ['user_id', 'movie_id'])

# Load a pickle data set that contain the train set for the ridge regression
test_pred = pd.read_pickle('data/test_pred.pickle')
test_pred = test_pred.rename(columns={"mf_rmse_rating": "MF_RMSE_rating"})


# Model used for the blending
models_names = [ 'MF_ALS_rating',
                'cocluster_rating',
                'knnmeans_item_rating',
                'knnmeans_user_rating',
                'knnzscore_user_rating',
                'knnzscore_item_rating',
                'knnbasic_user_rating',
                'knnbasic_item_rating',
                'slopeone_rating',
                'mf_rating',
                'svd_rating',
                'svdpp_rating',
                'item_mean_rating',
                'user_mean_rating',
                'global_mean_rating',
                'MF_RMSE_rating',
                'NN_deep_rating',
                'NN_shallow_rating']


# Cross validation on a ridge regression with ten fold
cv_ridge = KFold(n_splits=10)
gs_ridge = RidgeCV(alphas = [10**-i for i in range (-5, 10)], fit_intercept = False, scoring = 'neg_mean_squared_error', cv = cv_ridge)

# Fit the ridge regression
gs_ridge.fit(test_pred[models_names],test_pred['rating'] )

print ("Best lambda :", gs_ridge.alpha_, "\n")
print ("Optimal weight:", gs_ridge.coef_, "\n")

# Make predictions on the submission set
submission_pred['ridge_rating'] = gs_ridge.predict(submission_pred[models_names])

# Create a csv file
submission_path = 'results.csv'
create_csv(submission_path, submission_pred)

print("The submission file was completed successfully as 'result.csv'")