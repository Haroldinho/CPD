
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
from hyperopt import hp
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

# comptute mean_squared_error
rmsle_baseline = np.sqrt(mean_squared_log_error(y_test, baseline_predictions))

print("Baseline RMSE is {:.2f}".format(rmse_baseline))


# %% LOAD Data into DMATRIX for xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# %% XGBOOST Params
params = {
    # Parameters that we are going to tune.
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': .3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
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
print("Best RMSE: {:.2f} at iteration {:}.".format(model.best_score,
                                                   model.best_iteration))
# %% XGBoost cross-validation
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=88,
    nfold=5,
    metrics={'rmse'},
    early_stopping_rounds=10
)
cv_results
# %% Use Hyperopt
######################################
# Optimizing hyper parameters
######################################

X_train2, X_validation, y_train2, y_validation = train_test_split(X_train, y_train,
                                                                  test_size=.2, random_state=88)
dtrain2 = xgb.DMatrix(X_train2, label=y_train2)
dvalid = xgb.DMatrix(X_validation, label=y_validation)
#################################
# Define the search space
#################################
space = {
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'gamma': hp.uniform('gamma', 0, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    'seed': 88
}

################################
# Optimize max_depth and min_child_weight
#################################
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(5, 14)
    for min_child_weight in range(1, 10)
]
# Run cross validation on each pair
# Define initial best params and rmse
min_rmse = float('inf')
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight".format(
        max_depth, min_child_weight
    ))
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=88,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best RMSE
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(
        mean_rmse, boost_rounds
    ))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth, min_child_weight)
print("Best params: {}, {}, RMSE: {}".format(
    best_params[0], best_params[1], min_rmse
))

# Update the score
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]
# %% SAME with subsample and colsample_bytree
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i / 10. for i in range(5, 11)]
    for colsample in [i / 10. for i in range(6, 11)]
]
for subsample, colsample in gridsearch_params:
    print("CV with subsample={}, colsample".format(
        subsample, colsample
    ))
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=88,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best RMSE
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(
        mean_rmse, boost_rounds
    ))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample, colsample)
print("Best params: {}, {}, RMSE: {}".format(
    best_params[0], best_params[1], min_rmse
))

# Update the score
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]
params['eta'] = 0.1
# %%
#########################################
# Final Model
##########################################################
xgbr_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best RMSE: {:.3f} at iteration {:}.".format(model.best_score,
                                                   model.best_iteration))

y_pred = xgbr_model.predict(dtrain)
rmse = np.sqrt(mean_squared_error(y_pred, y_train))
print("Train set RMSE: {:.3f}".format(rmse))

y_pred2 = xgbr_model.predict(dtrain2)
rmse2 = np.sqrt(mean_squared_error(y_pred2, y_train2))
print("Train2 set RMSE: {:.3f}".format(rmse2))

y_pred = xgbr_model.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Test set RMSE: {:.3f}".format(rmse))
# %%
print("The best hyperparameters are : ", "\n")
print(params)
# %% SAVE THE MODEL
xgbr_model.save_model("condtional_change_xgboost_model.model")
# %%
# Simple sanity check to make sure everything makes sense
# Pick a few rows,
# Create the X matrix and predict on it

sample_df = change_prob_df.sample(10)
x_regressor = sample_df[[
    "Batch Size",
    "rho",
    "delta_rho",
    "Run Length",
    "A+",
    "A-",
    "Q+",
    "Q-",
    "W+",
    "W-"
]].values
deval = xgb.DMatrix(x_regressor)
y_predicted = xgbr_model.predict(deval)
saturated_y_predicted = np.where(y_predicted > 1,
                                 1, y_predicted)
sample_df["predicted"] = np.where(saturated_y_predicted < 0, 0,
                                  saturated_y_predicted).tolist()
sample_df.head()

# %%
# Using original model
y_predicted = model.predict(deval)
saturated_y_predicted = np.where(y_predicted > 1,
                                 1, y_predicted)
sample_df["predicted"] = np.where(saturated_y_predicted < 0, 0,
                                  saturated_y_predicted).tolist()
sample_df.head()
# %%
# THE NEW MODEL PERFORMS MUCH BETTER THAN THE ORIGINAL. SUCCESS!!!
