import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from helpers import create_csv

%load_ext autoreload
%autoreload 2


test_pred = pd.read_pickle('data/test_pred.pickle')
submission_pred = pd.read_pickle('data/submission_pred.pickle')

test_pred = test_pred.rename(columns={"mf_rmse_rating": "MF_RMSE_rating"})

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

cv_ridge = KFold(n_splits=10)
gs_ridge = RidgeCV(alphas = [10**-i for i in range (-5, 10)], fit_intercept = False, scoring = 'neg_mean_squared_error', cv = cv_ridge)


gs_ridge.fit (test_pred[models_names],test_pred['rating'] )

print ("Best lambda :", gs_ridge.alpha_, "\n")
print ("Optimal weight:", gs_ridge.coef_, "\n")


submission_pred['ridge_rating'] = gs_ridge.predict(submission_pred[models_names])

submission_path = 'results.csv'
create_csv(submission_path, submission_pred)