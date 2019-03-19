#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:35:50 2019

@author: jasond
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import fmin
import ast
from hyperopt import Trials
import lightgbm as lgb
import preprocessing as pp
import feature_selection as fs

MAX_EVALS = 500
N_FOLDS = 10

df_total = pd.read_csv('train.csv', index_col = None, header = 0, memory_map = True)
df_total = df_total.drop(['ID_code'],axis = 1)
#df_total = df_total.sample(1000)
#df_total.index = range(len(df_total))
frame_train,frame_test = pp.train_test_split(df_total, 'target', 0.3)
frame_train = pp.normalization(frame_train,'target')
frame_test = pp.normalization(frame_test,'target')
X = frame_train.drop(['target'], axis = 1)
y = frame_train['target']
X_pred = frame_test.drop(['target'],axis = 1)
y_truth = frame_test['target']
X = np.array(X)
X_pred = np.array(X_pred)
train_set = lgb.Dataset(X, label = y)
def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    
    run_time = timer() - start
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}
    
# Create the learning rate


# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'subsample': hp.uniform('gdbt_subsample', 0.5, 1), 
    'num_leaves': hp.quniform('num_leaves', 3, 13, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.007), np.log(0.01)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

tpe_algorithm = tpe.suggest
bayes_trials = Trials()
#Write CSV
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

results = pd.read_csv('gbm_trials.csv')
