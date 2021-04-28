"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: post_processing_joint_probabilities.py
Description: Library to process joint probabilites to get conditional probability of a false positives
which lead to significance levels and powers of the tests (Age, Queue Lenth, Waiting-times0
Author: Harold Nemo Adodo Nikoue
part of my partial observability thesis
Date: 4/28/2021
"""
from os import listdir, getcwd, chdir
from os.path import isfile, join

import pandas as pd

folder_name = "./Results/GLRT_ROSS/Performance_Tests/Hypothesis_Conditioned_on_Change/"

all_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
print(all_files)

orig_dir = getcwd()
chdir(folder_name)
# concatenate all conditional data
cond_prob_df_list = []
for file_name in all_files:
    local_df = pd.read_csv(file_name)
    cond_prob_df_list.append(local_df)
combined_cond_prob_df = pd.concat(cond_prob_df_list, axis=0)
combined_cond_prob_df.info()

###########################################
# Combine probabilities
###########################################

# 1. Two way probabilities
combined_cond_prob_df["A+, Q+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"] + \
                                           combined_cond_prob_df["A+, Q+, W- | Change"]
combined_cond_prob_df["A+, Q- | Change"] = combined_cond_prob_df["A+, Q-, W+ | Change"] + \
                                           combined_cond_prob_df["A+, Q-, W- | Change"]
combined_cond_prob_df["A+, W+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"] + \
                                           combined_cond_prob_df["A+, Q-, W+ | Change"]
combined_cond_prob_df["A+, W- | Change"] = combined_cond_prob_df["A+, Q+, W- | Change"] + \
                                           combined_cond_prob_df["A+, Q-, W- | Change"]
combined_cond_prob_df["Q+, W+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"] + \
                                           combined_cond_prob_df["A-, Q+, W+ | Change"]
combined_cond_prob_df["Q+, W- | Change"] = combined_cond_prob_df["A+, Q+, W- | Change"] + \
                                           combined_cond_prob_df["A-, Q+, W- | Change"]
combined_cond_prob_df["A-, Q+ | Change"] = combined_cond_prob_df["A-, Q+, W+ | Change"] + \
                                           combined_cond_prob_df["A-, Q+, W- | Change"]
combined_cond_prob_df["A-, Q- | Change"] = combined_cond_prob_df["A-, Q-, W+ | Change"] + \
                                           combined_cond_prob_df["A-, Q-, W- | Change"]
combined_cond_prob_df["A-, W+ | Change"] = combined_cond_prob_df["A-, Q+, W+ | Change"] + \
                                           combined_cond_prob_df["A-, Q-, W+ | Change"]
combined_cond_prob_df["A-, W- | Change"] = combined_cond_prob_df["A-, Q+, W- | Change"] + \
                                           combined_cond_prob_df["A-, Q-, W- | Change"]
combined_cond_prob_df["Q-, W+ | Change"] = combined_cond_prob_df["A+, Q-, W+ | Change"] + \
                                           combined_cond_prob_df["A-, Q-, W+ | Change"]
combined_cond_prob_df["Q-, W- | Change"] = combined_cond_prob_df["A+, Q-, W- | Change"] + \
                                           combined_cond_prob_df["A-, Q-, W- | Change"]

combined_cond_prob_df["A+, Q+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"] + \
                                              combined_cond_prob_df["A+, Q+, W- | No Change"]
combined_cond_prob_df["A+, Q- | No Change"] = combined_cond_prob_df["A+, Q-, W+ | No Change"] + \
                                              combined_cond_prob_df["A+, Q-, W- | No Change"]
combined_cond_prob_df["A+, W+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"] + \
                                              combined_cond_prob_df["A+, Q-, W+ | No Change"]
combined_cond_prob_df["A+, W- | No Change"] = combined_cond_prob_df["A+, Q+, W- | No Change"] + \
                                              combined_cond_prob_df["A+, Q-, W- | No Change"]
combined_cond_prob_df["Q+, W+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"] + \
                                              combined_cond_prob_df["A-, Q+, W+ | No Change"]
combined_cond_prob_df["Q+, W- | No Change"] = combined_cond_prob_df["A+, Q+, W- | No Change"] + \
                                              combined_cond_prob_df["A-, Q+, W- | No Change"]
combined_cond_prob_df["A-, Q+ | No Change"] = combined_cond_prob_df["A-, Q+, W+ | No Change"] + \
                                              combined_cond_prob_df["A-, Q+, W- | No Change"]
combined_cond_prob_df["A-, Q- | No Change"] = combined_cond_prob_df["A-, Q-, W+ | No Change"] + \
                                              combined_cond_prob_df["A-, Q-, W- | No Change"]
combined_cond_prob_df["A-, W+ | No Change"] = combined_cond_prob_df["A-, Q+, W+ | No Change"] + \
                                              combined_cond_prob_df["A-, Q-, W+ | No Change"]
combined_cond_prob_df["A-, W- | No Change"] = combined_cond_prob_df["A-, Q+, W- | No Change"] + \
                                              combined_cond_prob_df["A-, Q-, W- | No Change"]
combined_cond_prob_df["Q-, W+ | No Change"] = combined_cond_prob_df["A+, Q-, W+ | No Change"] + \
                                              combined_cond_prob_df["A-, Q-, W+ | No Change"]
combined_cond_prob_df["Q-, W- | No Change"] = combined_cond_prob_df["A+, Q-, W- | No Change"] + \
                                              combined_cond_prob_df["A-, Q-, W- | No Change"]
# 2. one way probability
combined_cond_prob_df["A+ | Change"] = combined_cond_prob_df["A+, Q+ | Change"] + \
                                       combined_cond_prob_df["A+, Q- | Change"] + \
                                       combined_cond_prob_df["A+, W+ | Change"] + \
                                       combined_cond_prob_df["A+, W- | Change"]
combined_cond_prob_df["A+ | No Change"] = combined_cond_prob_df["A+, Q+ | No Change"] + \
                                          combined_cond_prob_df["A+, Q- | No Change"] + \
                                          combined_cond_prob_df["A+, W+ | No Change"] + \
                                          combined_cond_prob_df["A+, W- | No Change"]
combined_cond_prob_df["A- | Change"] = combined_cond_prob_df["A-, Q+ | Change"] + \
                                       combined_cond_prob_df["A-, Q- | Change"] + \
                                       combined_cond_prob_df["A-, W+ | Change"] + \
                                       combined_cond_prob_df["A-, W- | Change"]
combined_cond_prob_df["A- | No Change"] = combined_cond_prob_df["A-, Q+ | No Change"] + \
                                          combined_cond_prob_df["A-, Q- | No Change"] + \
                                          combined_cond_prob_df["A-, W+ | No Change"] + \
                                          combined_cond_prob_df["A-, W- | No Change"]
combined_cond_prob_df["Q+ | Change"] = combined_cond_prob_df["A+, Q+ | Change"] + \
                                       combined_cond_prob_df["A-, Q+ | Change"] + \
                                       combined_cond_prob_df["Q+, W+ | Change"] + \
                                       combined_cond_prob_df["Q+, W- | Change"]
combined_cond_prob_df["Q+ | No Change"] = combined_cond_prob_df["A+, Q+ | No Change"] + \
                                          combined_cond_prob_df["A-, Q+ | No Change"] + \
                                          combined_cond_prob_df["Q+, W+ | No Change"] + \
                                          combined_cond_prob_df["Q+, W- | No Change"]
combined_cond_prob_df["Q- | Change"] = combined_cond_prob_df["A+, Q- | Change"] + \
                                       combined_cond_prob_df["A-, Q- | Change"] + \
                                       combined_cond_prob_df["Q-, W+ | Change"] + \
                                       combined_cond_prob_df["Q-, W- | Change"]
combined_cond_prob_df["Q- | No Change"] = combined_cond_prob_df["A+, Q- | No Change"] + \
                                          combined_cond_prob_df["A-, Q- | No Change"] + \
                                          combined_cond_prob_df["Q-, W+ | No Change"] + \
                                          combined_cond_prob_df["Q-, W- | No Change"]
combined_cond_prob_df["W+ | Change"] = combined_cond_prob_df["A+, W+ | Change"] + \
                                       combined_cond_prob_df["A-, W+ | Change"] + \
                                       combined_cond_prob_df["Q+, W+ | Change"] + \
                                       combined_cond_prob_df["Q-, W+ | Change"]
combined_cond_prob_df["W+ | No Change"] = combined_cond_prob_df["A+, W+ | No Change"] + \
                                          combined_cond_prob_df["A-, W+ | No Change"] + \
                                          combined_cond_prob_df["Q+, W+ | No Change"] + \
                                          combined_cond_prob_df["Q-, W+ | No Change"]
combined_cond_prob_df["W- | Change"] = combined_cond_prob_df["A+, W- | Change"] + \
                                       combined_cond_prob_df["A-, W- | Change"] + \
                                       combined_cond_prob_df["Q+, W- | Change"] + \
                                       combined_cond_prob_df["Q-, W- | Change"]
combined_cond_prob_df["W- | No Change"] = combined_cond_prob_df["A+, W- | No Change"] + \
                                          combined_cond_prob_df["A-, W- | No Change"] + \
                                          combined_cond_prob_df["Q+, W- | No Change"] + \
                                          combined_cond_prob_df["Q-, W- | No Change"]
# save file
combined_cond_prob_df.to_csv("../joint_conditional_probability.csv")

# return to original path
chdir(orig_dir)
