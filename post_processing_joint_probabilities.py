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

import numpy as np
import pandas as pd


def combine_changes_conditioned_on_hypothesis(folder_name):
    all_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    orig_dir = getcwd()
    chdir(folder_name)
    # concatenate all conditional data
    cond_prob_df_list = []
    for file_name in all_files:
        local_df = pd.read_csv(file_name)
        cond_prob_df_list.append(local_df)
    combined_cond_prob_df = pd.concat(cond_prob_df_list, axis=0)
    combined_cond_prob_df.info()
    combined_cond_prob_df.to_csv("../joint_conditional_probability_change_conditioned_on_hypothesis.csv")
    chdir(orig_dir)


def combine_hypothesis_conditioned_on_change(folder_name):
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
    final_cond_prob_df = pd.DataFrame()

    ###########################################
    # Combine probabilities
    ###########################################

    # 0. Copy all relevant data
    final_cond_prob_df["Run Length"] = combined_cond_prob_df["Run Length"]
    final_cond_prob_df["rho"] = combined_cond_prob_df["rho"]
    final_cond_prob_df["delta_rho"] = combined_cond_prob_df["delta_rho"]
    final_cond_prob_df["Batch Size"] = combined_cond_prob_df["Batch Size"]

    final_cond_prob_df["A+, Q+, W+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"]
    final_cond_prob_df["A-, Q-, W- | Change"] = combined_cond_prob_df["A-, Q-, W- | Change"]
    final_cond_prob_df["A+, Q-, W- | Change"] = combined_cond_prob_df["A+, Q-, W- | Change"]
    final_cond_prob_df["A+, Q+, W- | Change"] = combined_cond_prob_df["A+, Q+, W- | Change"]
    final_cond_prob_df["A-, Q+, W- | Change"] = combined_cond_prob_df["A-, Q+, W- | Change"]
    final_cond_prob_df["A-, Q-, W+ | Change"] = combined_cond_prob_df["A-, Q-, W+ | Change"]
    final_cond_prob_df["A+, Q-, W+ | Change"] = combined_cond_prob_df["A+, Q-, W+ | Change"]
    final_cond_prob_df["A-, Q+, W+ | Change"] = combined_cond_prob_df["A-, Q+, W+ | Change"]

    final_cond_prob_df["A+, Q+, W+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"]
    final_cond_prob_df["A-, Q-, W- | No Change"] = combined_cond_prob_df["A-, Q-, W- | No Change"]
    final_cond_prob_df["A+, Q-, W- | No Change"] = combined_cond_prob_df["A+, Q-, W- | No Change"]
    final_cond_prob_df["A+, Q+, W- | No Change"] = combined_cond_prob_df["A+, Q+, W- | No Change"]
    final_cond_prob_df["A-, Q+, W- | No Change"] = combined_cond_prob_df["A-, Q+, W- | No Change"]
    final_cond_prob_df["A-, Q-, W+ | No Change"] = combined_cond_prob_df["A-, Q-, W+ | No Change"]
    final_cond_prob_df["A+, Q-, W+ | No Change"] = combined_cond_prob_df["A+, Q-, W+ | No Change"]
    final_cond_prob_df["A-, Q+, W+ | No Change"] = combined_cond_prob_df["A-, Q+, W+ | No Change"]
    # 1. Two way probabilities
    final_cond_prob_df["A+, Q+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"] + \
                                            combined_cond_prob_df["A+, Q+, W- | Change"]
    final_cond_prob_df["A+, Q- | Change"] = combined_cond_prob_df["A+, Q-, W+ | Change"] + \
                                            combined_cond_prob_df["A+, Q-, W- | Change"]
    final_cond_prob_df["A+, W+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"] + \
                                            combined_cond_prob_df["A+, Q-, W+ | Change"]
    final_cond_prob_df["A+, W- | Change"] = combined_cond_prob_df["A+, Q+, W- | Change"] + \
                                            combined_cond_prob_df["A+, Q-, W- | Change"]
    # 2. one way probability
    final_cond_prob_df["A+ | Change"] = np.maximum(
        final_cond_prob_df["A+, Q+ | Change"] + final_cond_prob_df["A+, Q- | Change"],
        final_cond_prob_df["A+, W+ | Change"] + final_cond_prob_df["A+, W- | Change"])

    final_cond_prob_df["Q+, W+ | Change"] = combined_cond_prob_df["A+, Q+, W+ | Change"] + \
                                            combined_cond_prob_df["A-, Q+, W+ | Change"]
    final_cond_prob_df["Q+, W- | Change"] = combined_cond_prob_df["A+, Q+, W- | Change"] + \
                                            combined_cond_prob_df["A-, Q+, W- | Change"]
    final_cond_prob_df["A-, Q+ | Change"] = combined_cond_prob_df["A-, Q+, W+ | Change"] + \
                                            combined_cond_prob_df["A-, Q+, W- | Change"]
    final_cond_prob_df["Q+ | Change"] = np.maximum(
        final_cond_prob_df["A+, Q+ | Change"] + final_cond_prob_df["A-, Q+ | Change"],
        final_cond_prob_df["Q+, W+ | Change"] + final_cond_prob_df["Q+, W- | Change"])

    final_cond_prob_df["A-, Q- | Change"] = combined_cond_prob_df["A-, Q-, W+ | Change"] + \
                                            combined_cond_prob_df["A-, Q-, W- | Change"]
    final_cond_prob_df["A-, W+ | Change"] = combined_cond_prob_df["A-, Q+, W+ | Change"] + \
                                            combined_cond_prob_df["A-, Q-, W+ | Change"]
    final_cond_prob_df["A-, W- | Change"] = combined_cond_prob_df["A-, Q+, W- | Change"] + \
                                            combined_cond_prob_df["A-, Q-, W- | Change"]
    final_cond_prob_df["A- | Change"] = np.maximum(
        final_cond_prob_df["A-, Q+ | Change"] + final_cond_prob_df["A-, Q- | Change"],
        final_cond_prob_df["A-, W+ | Change"] + final_cond_prob_df["A-, W- | Change"])

    final_cond_prob_df["Q-, W+ | Change"] = combined_cond_prob_df["A+, Q-, W+ | Change"] + \
                                            combined_cond_prob_df["A-, Q-, W+ | Change"]
    final_cond_prob_df["Q-, W- | Change"] = combined_cond_prob_df["A+, Q-, W- | Change"] + \
                                            combined_cond_prob_df["A-, Q-, W- | Change"]
    final_cond_prob_df["Q- | Change"] = np.maximum(
        final_cond_prob_df["A+, Q- | Change"] + final_cond_prob_df["A-, Q- | Change"],
        final_cond_prob_df["Q-, W+ | Change"] + final_cond_prob_df["Q-, W- | Change"])
    final_cond_prob_df["W- | Change"] = np.maximum(
        final_cond_prob_df["A+, W- | Change"] + final_cond_prob_df["A-, W- | Change"],
        final_cond_prob_df["Q+, W- | Change"] + final_cond_prob_df["Q-, W- | Change"])
    final_cond_prob_df["W+ | Change"] = np.maximum(
        final_cond_prob_df["A+, W+ | Change"] + final_cond_prob_df["A-, W+ | Change"],
        final_cond_prob_df["Q+, W+ | Change"] + final_cond_prob_df["Q-, W+ | Change"])

    final_cond_prob_df["A+, Q+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"] + \
                                               combined_cond_prob_df["A+, Q+, W- | No Change"]
    final_cond_prob_df["A+, Q- | No Change"] = combined_cond_prob_df["A+, Q-, W+ | No Change"] + \
                                               combined_cond_prob_df["A+, Q-, W- | No Change"]
    final_cond_prob_df["A+, W+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"] + \
                                               combined_cond_prob_df["A+, Q-, W+ | No Change"]
    final_cond_prob_df["A+, W- | No Change"] = combined_cond_prob_df["A+, Q+, W- | No Change"] + \
                                               combined_cond_prob_df["A+, Q-, W- | No Change"]
    final_cond_prob_df["Q+, W+ | No Change"] = combined_cond_prob_df["A+, Q+, W+ | No Change"] + \
                                               combined_cond_prob_df["A-, Q+, W+ | No Change"]
    final_cond_prob_df["Q+, W- | No Change"] = combined_cond_prob_df["A+, Q+, W- | No Change"] + \
                                               combined_cond_prob_df["A-, Q+, W- | No Change"]
    final_cond_prob_df["A-, Q+ | No Change"] = combined_cond_prob_df["A-, Q+, W+ | No Change"] + \
                                               combined_cond_prob_df["A-, Q+, W- | No Change"]
    final_cond_prob_df["A-, Q- | No Change"] = combined_cond_prob_df["A-, Q-, W+ | No Change"] + \
                                               combined_cond_prob_df["A-, Q-, W- | No Change"]
    final_cond_prob_df["A-, W+ | No Change"] = combined_cond_prob_df["A-, Q+, W+ | No Change"] + \
                                               combined_cond_prob_df["A-, Q-, W+ | No Change"]
    final_cond_prob_df["A-, W- | No Change"] = combined_cond_prob_df["A-, Q+, W- | No Change"] + \
                                               combined_cond_prob_df["A-, Q-, W- | No Change"]
    final_cond_prob_df["Q-, W+ | No Change"] = combined_cond_prob_df["A+, Q-, W+ | No Change"] + \
                                               combined_cond_prob_df["A-, Q-, W+ | No Change"]
    final_cond_prob_df["Q-, W- | No Change"] = combined_cond_prob_df["A+, Q-, W- | No Change"] + \
                                               combined_cond_prob_df["A-, Q-, W- | No Change"]

    final_cond_prob_df["A+ | No Change"] = np.maximum(final_cond_prob_df["A+, Q+ | No Change"] +
                                                      final_cond_prob_df["A+, Q- | No Change"],
                                                      final_cond_prob_df["A+, W+ | No Change"] +
                                                      final_cond_prob_df["A+, W- | No Change"])
    final_cond_prob_df["A- | No Change"] = np.maximum(final_cond_prob_df["A-, Q+ | No Change"] + \
                                                      final_cond_prob_df["A-, Q- | No Change"],
                                                      final_cond_prob_df["A-, W+ | No Change"] + \
                                                      final_cond_prob_df["A-, W- | No Change"])
    final_cond_prob_df["Q+ | No Change"] = np.maximum(final_cond_prob_df["A+, Q+ | No Change"] + \
                                                      final_cond_prob_df["A-, Q+ | No Change"],
                                                      final_cond_prob_df["Q+, W+ | No Change"] + \
                                                      final_cond_prob_df["Q+, W- | No Change"])
    final_cond_prob_df["Q- | No Change"] = np.maximum(final_cond_prob_df["A+, Q- | No Change"] +
                                                      final_cond_prob_df["A-, Q- | No Change"],
                                                      final_cond_prob_df["Q-, W+ | No Change"] + \
                                                      final_cond_prob_df["Q-, W- | No Change"])
    final_cond_prob_df["W+ | No Change"] = np.maximum(final_cond_prob_df["A+, W+ | No Change"] + \
                                                      final_cond_prob_df["A-, W+ | No Change"],
                                                      final_cond_prob_df["Q+, W+ | No Change"] + \
                                                      final_cond_prob_df["Q-, W+ | No Change"])
    final_cond_prob_df["W- | No Change"] = np.maximum(final_cond_prob_df["A+, W- | No Change"] + \
                                                      final_cond_prob_df["A-, W- | No Change"],
                                                      final_cond_prob_df["Q+, W- | No Change"] + \
                                                      final_cond_prob_df["Q-, W- | No Change"])

    final_cond_prob_df["A | Change"] = final_cond_prob_df["A+ | Change"] + final_cond_prob_df["A- | Change"]
    final_cond_prob_df["Q | Change"] = final_cond_prob_df["Q+ | Change"] + final_cond_prob_df["Q- | Change"]
    final_cond_prob_df["W | Change"] = final_cond_prob_df["W+ | Change"] + final_cond_prob_df["W- | Change"]
    final_cond_prob_df["A | No Change"] = final_cond_prob_df["A+ | No Change"] + final_cond_prob_df["A- | No Change"]
    final_cond_prob_df["Q | No Change"] = final_cond_prob_df["Q+ | No Change"] + final_cond_prob_df["Q- | No Change"]
    final_cond_prob_df["W | No Change"] = final_cond_prob_df["W+ | No Change"] + final_cond_prob_df["W- | No Change"]

    # assert((final_cond_prob_df["A+ | Change"] + final_cond_prob_df["A- | Change"] <= 1.01).all())
    # save file
    final_cond_prob_df.to_csv("../joint_conditional_probability_hypothesis_conditioned_on_change.csv")

    # return to original path
    chdir(orig_dir)


def invert_conditional_probability_direction():
    """
    Go from hypothesis given a change to change given a hypothesis
    """
    p_change = 1 - np.exp(-1)
    p_no_change = 1 - p_change
    cond_hypothesis_given_change_df = pd.read_csv(
        "./Results/GLRT_ROSS/Performance_Tests/joint_conditional_probability_3.csv")

    # Make the transformation
    cond_change_given_hypothesis_df = cond_hypothesis_given_change_df[["rho", "delta_rho", "Batch Size"]].copy()
    cond_change_given_hypothesis_df["Change | A+"] = \
        cond_hypothesis_given_change_df["A+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-"] = \
        cond_hypothesis_given_change_df["A- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A- | Change"] * p_change
                -
                cond_hypothesis_given_change_df["A- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | Q+"] = \
        cond_hypothesis_given_change_df["Q+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["Q+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["Q+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | Q-"] = \
        cond_hypothesis_given_change_df["Q- | Change"] * p_change / (
                cond_hypothesis_given_change_df["Q- | Change"] * p_change
                -
                cond_hypothesis_given_change_df["Q- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | W+"] = \
        cond_hypothesis_given_change_df["W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | W-"] = \
        cond_hypothesis_given_change_df["W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["W- | Change"] * p_change
                -
                cond_hypothesis_given_change_df["W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, Q+"] = \
        cond_hypothesis_given_change_df["A+, Q+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, Q+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, Q+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, Q-"] = \
        cond_hypothesis_given_change_df["A+, Q- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, Q- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, Q- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, W+"] = \
        cond_hypothesis_given_change_df["A+, W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, W-"] = \
        cond_hypothesis_given_change_df["A+, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | Q+, W+"] = \
        cond_hypothesis_given_change_df["Q+, W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["Q+, W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["Q+, W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | Q+, W-"] = \
        cond_hypothesis_given_change_df["Q+, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["Q+, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["Q+, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, Q+"] = \
        cond_hypothesis_given_change_df["A-, Q+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, Q+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, Q+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, Q-"] = \
        cond_hypothesis_given_change_df["A-, Q- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, Q- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, Q- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, W+"] = \
        cond_hypothesis_given_change_df["A-, W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, W-"] = \
        cond_hypothesis_given_change_df["A-, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | Q-, W+"] = \
        cond_hypothesis_given_change_df["Q-, W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["Q-, W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["Q-, W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | Q-, W-"] = \
        cond_hypothesis_given_change_df["Q-, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["Q-, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["Q-, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, Q+, W+"] = \
        cond_hypothesis_given_change_df["A+, Q+, W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, Q+, W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, Q+, W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, Q-, W-"] = \
        cond_hypothesis_given_change_df["A-, Q-, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, Q-, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, Q-, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, Q-, W-"] = \
        cond_hypothesis_given_change_df["A+, Q-, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, Q-, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, Q-, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A+, Q+, W-"] = \
        cond_hypothesis_given_change_df["A+, Q+, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A+, Q+, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A+, Q+, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, Q+, W-"] = \
        cond_hypothesis_given_change_df["A-, Q+, W- | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, Q+, W- | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, Q+, W- | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df["Change | A-, Q-, W+"] = \
        cond_hypothesis_given_change_df["A-, Q-, W+ | Change"] * p_change / (
                cond_hypothesis_given_change_df["A-, Q-, W+ | Change"] * p_change
                +
                cond_hypothesis_given_change_df["A-, Q-, W+ | No Change"] * p_no_change
        )
    cond_change_given_hypothesis_df.to_csv("./Results/GLRT_ROSS/Performance_Tests/" +
                                           "joint_conditional_probability_hypothesis_given_change.csv")


if __name__ == "__main__":
    folder_name_ab = "./Results/GLRT_ROSS/Performance_Tests/Hypothesis_Conditioned_on_Change/"
    combine_hypothesis_conditioned_on_change(folder_name_ab)
#     folder_name_ba = "./Results/GLRT_ROSS/Performance_Tests/ChangePoint_Conditioned_on_HypothesisOutcome/"
#     combine_changes_conditioned_on_hypothesis(folder_name_ba)
# Compute the inverted conditional probability
#     invert_conditional_probability_direction()
