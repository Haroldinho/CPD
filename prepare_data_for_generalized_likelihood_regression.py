"""

            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: prepare_data_for_generalized_likelihood_regression.py
Description:
    Prepare data for generalize_likelihood_of_a_change_for_parallel_tests.py
    Retain only the columns of interest and perform the one-hot encoding on the hypothesis combination
    A? | Q? | W?
supports main_tests.py
Author: Harold Nemo Adodo Nikoue
part of my chapter on parallel partial observability in my thesis
Date: 10/16/2021
"""

import re

# %%
import pandas as pd

# %%
# PREPROCESSING
#
# see if I can load the file
folder_name = "Results/GLRT_ROSS/Performance_Tests/"
orig_likelihood_df = pd.read_csv(folder_name + "joint_conditional_probability_change_conditioned_on_hypothesis.csv")

# print(orig_likelihood_df.info())
# %%
## TAKE THE COLUMNS I CARE ABOUT
## For the 3 way joint distribution
likelihood_3joint_obs_df = orig_likelihood_df[[
    "Batch Size", "rho", "delta_rho", "Run Length",
    "Change | A+, Q+, W+",
    "Change | A+, Q+, W-",
    "Change | A+, Q-, W+",
    "Change | A-, Q+, W+",
    "Change | A-, Q+, W-",
    "Change | A-, Q-, W+",
    "Change | A+, Q-, W-",
    "Change | A-, Q-, W-",
]]
column_names = list(likelihood_3joint_obs_df.columns)

# %%
from typing import List, Dict


def transform_hypothesis_encoding_to_one_hot_code(hypothesis: str) -> Dict:
    if not re.search("A|W|Q", hypothesis):
        return None
    coded_hypothesis = dict()
    if re.search("A\+", hypothesis):
        coded_hypothesis["A+"] = 1
    else:
        coded_hypothesis["A+"] = 0

    if re.search("A\-", hypothesis):
        coded_hypothesis["A-"] = 1
    else:
        coded_hypothesis["A-"] = 0

    if re.search("Q\+", hypothesis):
        coded_hypothesis["Q+"] = 1
    else:
        coded_hypothesis["Q+"] = 0

    if re.search("Q\-", hypothesis):
        coded_hypothesis["Q-"] = 1
    else:
        coded_hypothesis["Q-"] = 0

    if re.search("W\+", hypothesis):
        coded_hypothesis["W+"] = 1
    else:
        coded_hypothesis["W+"] = 0

    if re.search("W\-", hypothesis):
        coded_hypothesis["W-"] = 1
    else:
        coded_hypothesis["W-"] = 0

    return coded_hypothesis


# %%
# Create an extended dictionary by repeating each entry n times
def extend_dict_for_n_rows(hypothesis_dic: List, num_rows: int) -> List:
    new_dic = {}
    for key, val in hypothesis_dic.items():
        new_dic[key] = [val] * num_rows
    return new_dic


def create_extended_dict_from_hypothesis(hypothesis: str, num_rows: int) -> List:
    hypothesis_dict = transform_hypothesis_encoding_to_one_hot_code(hypothesis)
    return extend_dict_for_n_rows(hypothesis_dict, num_rows)


def create_dataframe_from_hypothesis(hypothesis: str, num_rows: int) -> List:
    return pd.DataFrame(create_extended_dict_from_hypothesis(hypothesis, num_rows))


# %%
base_df = likelihood_3joint_obs_df[["Batch Size", "rho", "delta_rho", "Run Length"]]
hypothesis_names = column_names[5:]
full_df_frames = []
num_rows = len(base_df)
for hypothesis_string in hypothesis_names:
    if "No" in hypothesis_string:
        continue
    # get new dataframe
    hypothesis_df = create_dataframe_from_hypothesis(hypothesis_string, num_rows)
    new_df = pd.concat([
        base_df,
        likelihood_3joint_obs_df.rename(
            columns={hypothesis_string: "Change"}
        )["Change"],
        hypothesis_df
    ], axis=1)
    full_df_frames.append(new_df)
full_df = pd.concat(full_df_frames)
# %%
full_df.to_csv(folder_name + "joint_conditional_probability_change_conditioned_on_hypothesis_for_learning.csv",
               index=False)
# %%
## While training, I can replace the NaN by zeros
