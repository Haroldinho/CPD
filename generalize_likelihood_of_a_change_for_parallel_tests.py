
"""

            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: generalize_likelihood_of_a_change_for_parallel_tests.py
Description:
    Learn how to generalize the probabilities of 
    joint_conditional_probability_hypothesis_conditioned_on_change.xlsx
    Save the function as a pickle file that can be used for prediction
supports main_tests.py
Author: Harold Nemo Adodo Nikoue
part of my chapter on parallel partial observability in my thesis
Date: 10/17/2021
"""
import numpy as np
# %% IMPORTS
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split

folder_name = "Results/GLRT_ROSS/Performance_Tests/"
change_prob_file_name = folder_name + \
                        "joint_conditional_probability_change_conditioned_on_hypothesis_for_learning.csv"
# %%
change_prob_df = pd.read_csv(change_prob_file_name)

# %% TRAIN TEST split

target_df = change_prob_df["Change"]
regressor_df = change_prob_df.loc[:, change_prob_df.columns != 'Change']
X_matrix, y_matrix =  regressor_df.values, target_df.values
X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, 
test_size=.1, random_state=88)

#%% Building baseline
mean_train = np.mean(y_train)
baseline_predictions = np.ones(y_test.shape) * mean_train

# compute mean_squared_error
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_predictions))

# comptute mean_squared_log_error
rmsle_baseline = np.sqrt(mean_squared_log_error(y_test, baseline_predictions))

print("Baseline RMSLE is {:.2f}".format(rmsle_baseline))


# %% LOAD Data into DMATRIX for xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# %% XGBOOST Params
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:squaredlogerror',
    'eval_metric':'rmsle'
}
num_boost_round = 999


# %% XGBoost Model
model = xgb.train(
    params, 
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best RMSLE: {:.3f} with {} rounds".format(
    model.best_score,
    model.best_iteration+1
))
# %% XGBoost cross-validation
cv_results = xgb.cv(
    params,
    dtrain, 
    num_boost_round=num_boost_round,
    seed=88,
    nfold=5,
    metrics={'rmsle'},
    early_stopping_rounds=10
)
cv_results
# %% Use Hyperopt
######################################
# Optimizign hyper parameters
######################################

X_train2, X_validation, y_train2, y_validation = train_test_split(X_train, y_train, 
test_size=.1, random_state=88)
#################################
# Define the search space
#################################
space = {
    'max_depth': hp.quniform('max_depth', 3, 18, 1),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'subsample': hp.uniform('subsample', 0.7, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators':180,
    'seed':88
}

xgb_fit_params = {
    'eval_metric': 'rmsle',
    'early_stopping_rounds': 10,
    'verbose': False
}

xgb_para = dict()
xgb_para['reg_params'] = space
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_log_error(y, pred))
#################################
# Define the objective function
#################################
def objective(space):
    clf = xgb.XGBRegressor(
        n_estimators = space['n_estimators'],
        max_depth = int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        min_child_weight=int(space['min_child_weight']),
        reg_lambda=space['reg_lambda'],
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree']
    )
    evaluation = [(X_train2, y_train2), (X_validation, y_validation)]
    clf.fit(
        X_train2, y_train2,
        eval_set=evaluation,
        eval_metric='rmsle',
        early_stopping_rounds=10,
        verbose=False
    )
    pred=clf.predict(X_validation)
    score = np.sqrt(mean_squared_log_error(y_validation, pred))
    print("Validation score: {:.2f}".format(score))

#     #Using kfold cross validation
#     kfold = KFold(n_splits=5, shuffle=True)
#     kf_cv_scores = cross_val_score(clf, X_train2, y_train2, 
#     scoring='neg_mean_squared_log_error', cv=kfold )
    return {'loss': score, 'status': STATUS_OK}

# %%
trials = Trials()
best_hyperparams = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
# %%
# Predict test data
xgbr_best = xgb.XGBRegressor(
    n_estimators = 180,
    max_depth = int(best_hyperparams['max_depth']),
    gamma=best_hyperparams['gamma'],
    reg_alpha=int(best_hyperparams['reg_alpha']),
    min_child_weight=int(best_hyperparams['min_child_weight']),
    reg_lambda=best_hyperparams['reg_lambda'],
    subsample=best_hyperparams['subsample'],
    colsample_bytree=best_hyperparams['colsample_bytree']
)
evaluation = [(X_train, y_train), (X_test, y_test)]
xgbr_best.fit(
    X_train, y_train,
    eval_set=evaluation,
    eval_metric='rmsle',
    early_stopping_rounds=10,
    verbose=False
)
#%%
y_pred = xgbr_best.predict(X_train)
rmsle = mean_squared_log_error(y_pred, y_train)
print("Train set RMSLE: {:.2f}".format(rmsle))

y_pred2 = xgbr_best.predict(X_train2)
rmsle2 = mean_squared_log_error(y_pred2, y_train2)
print("Train2 set RMSLE: {:.2f}".format(rmsle2))

y_pred = xgbr_best.predict(X_test)
rmsle = mean_squared_log_error(y_pred, y_test)
print("Test set RMSLE: {:.2f}".format(rmsle))
# %%
print("The best hyperparameters are : ", "\n")
print(best_hyperparams)
# %% SAVE THE MODEL
xgbr_best.save_model("condtional_change_xgboost_model.model")
# %%
# Simple sanity check to make sure everything makes sense
# Pick a few rows,
# Create the X matrix and predict on it
