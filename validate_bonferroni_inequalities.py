"""

            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: validate_bonferroni_inequalities.py
Description: Moving from marginal probabilities to joint probabilities inequalities
Experience to see how probability inequalities can be used to bound the joint probabilities of unseen combinations
Different experiments will take place:
1) Count the number of joint probabilities that do not fall within the Bonferroni inequalities
2) Visualize the Bonferroni inequalities as multiple bands with the actual value in the middle hopefully

Will need to have the following functions
-- a function to read all the joint probabilities for different rhos and ms and return a dataframe
-- a function to compute the lower an upper bound given S1, S2, S3 and S4
-- a main method that goes over all the cases and perform the count
Author: Harold Nemo Adodo Nikoue
Date: 9/03/2021
"""
import pandas as pd
import numpy as np
import csv
import seaborn as sns
from re import search
from matplotlib import pyplot as plt


def return_joint_probabilities_df_given_change():
    """
    We want to read all the rows in "joint_conditional_probability_hypothesis_conditioned_on_change.csv"
    under Results/GLRT_ROSS/Performance_Tests/joint_conditional_probability_hypothesis_conditioned_on_change.csv
    :return: pandas DataFrame with all the joint probabilies conditioned on a change
    """
    file_name = "Results/GLRT_ROSS/Performance_Tests/" + \
                "joint_conditional_probability_hypothesis_conditioned_on_change.csv"
    return pd.read_csv(file_name)


def compute_union_of_events(p_a_1, p_a_2, p_a_3, p_a_12, p_a_13, p_a_23, p_a_123):
    """
    :parameter p_a_1: probability of event a_1
    :parameter p_a_2: probability of event a_2
    :parameter p_a_3: probability of event a_3
    :parameter p_a_12: probability of event a_12
    :parameter p_a_13: probability of event a_13
    :parameter p_a_23: probability of event a_23
    :parameter p_a_123: probability of event a_123
    :return: probability of the union of all 3 events
    """
    return p_a_1 + p_a_2 + p_a_3 - p_a_12 - p_a_13 - p_a_23 + p_a_123.values


def compute_lower_bound(s_1, s_2):
    return_series = 2 / 3 * s_1 - 1 / 3 * s_2
    return_series[return_series > 1.0] = 1.0 - 1e-1
    return return_series


def compute_upper_bound(s_1, s_2):
    return_series = s_1 - 2 / 3 * s_2
    return_series[return_series > 1.0] = 1.0 + 1e-1
    return return_series


def return_z_scen1_change(row):
    return float(row["A+ | Change"] + row["Q+ | Change"]) + float(row["W+ | Change"] - row["A+, Q+ | Change"]) \
           - float(row["A+, W+ | Change"] - row["Q+, W+ | Change"]) + row["A+, Q+, W+ | Change"]


def apply_union_of_events_to_df_given_change_scen_1(current_df):
    column_a = pd.to_numeric(current_df["A+ | Change"]) + pd.to_numeric(current_df["Q+ | Change"])
    column_b = column_a + pd.to_numeric(current_df["W+ | Change"])
    column_c = column_b - pd.to_numeric(current_df["A+, Q+ | Change"])
    column_d = column_c - pd.to_numeric(current_df["A+, W+ | Change"])
    column_e = column_d - pd.to_numeric(current_df["Q+, W+ | Change"])
    column_f = column_e + pd.to_numeric(current_df["A+, Q+, W+ | Change"])
    return column_f


def return_union_of_events_to_df_given_change_scen_1(current_df):
    """
    :param current_df: input probability dataframe
    :return: pd series for the union
    """
    #    column_a_1 = current_df["A+ | Change"]
    #    column_a_2 = current_df["Q+ | Change"]
    #    column_a_3 = current_df["W+ | Change"]
    #    column_a_12 = current_df["A+, Q+ | Change"]
    #    column_a_13 = current_df["A+, W+ | Change"]
    #    column_a_23 = current_df["Q+, W+ | Change"]
    #    column_a_123 = current_df["A+, Q+, W+ | Change"]
    #    lambda_fun = lambda row: row["A+ | Change"] + row["Q+ | Change"] + row["W+ | Change"] - row["A+, Q+ | Change"] \
    #                             - row["A+, W+ | Change"] - row["Q+, W+ | Change"] + row["A+, Q+, W+ | Change"]
    z = []
    for idx, row in current_df.iterrows():
        z.append(row["A+ | Change"] + row["Q+ | Change"] + row["W+ | Change"] - row["A+, Q+ | Change"]
                 - row["A+, W+ | Change"] - row["Q+, W+ | Change"] + row["A+, Q+, W+ | Change"])
    print(z)
    return z


def apply_union_of_events_to_df_given_no_change_scen_1(current_df):
    lambda_fun = lambda row: row["A+ | No Change"] + row["Q+ | No Change"] + row["W+ | No Change"] - row[
        "A+, Q+ | No Change"] \
                             - row["A+, W+ | No Change"] - row["Q+, W+ | No Change"] + row["A+, Q+, W+ | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_2(current_df):
    lambda_fun = lambda row: row["A- | Change"] + row["Q+ | Change"] + row["W+ | Change"] - row["A-, Q+ | Change"] \
                             - row["A-, W+ | Change"] - row["Q+, W+ | Change"] + row["A-, Q+, W+ | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_nochange_scen_2(current_df):
    lambda_fun = lambda row: row["A- | No Change"] + row["Q+ | No Change"] + row["W+ | No Change"] - row[
        "A-, Q+ | No Change"] \
                             - row["A-, W+ | No Change"] - row["Q+, W+ | No Change"] + row["A-, Q+, W+ | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_3(current_df):
    lambda_fun = lambda row: row["A+ | Change"] + row["Q- | Change"] + row["W+ | Change"] - row["A+, Q- | Change"] \
                             - row["A+, W+ | Change"] - row["Q-, W+ | Change"] + row["A+, Q-, W+ | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_no_change_scen_3(current_df):
    lambda_fun = lambda row: row["A+ | No Change"] + row["Q- | No Change"] + row["W+ | No Change"] - row[
        "A+, Q- | No Change"] \
                             - row["A+, W+ | No Change"] - row["Q-, W+ | No Change"] + row["A+, Q-, W+ | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_4(current_df):
    lambda_fun = lambda row: row["A+ | Change"] + row["Q+ | Change"] + row["W- | Change"] - row["A+, Q+ | Change"] \
                             - row["A+, W- | Change"] - row["Q+, W- | Change"] + row["A+, Q+, W- | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_no_change_scen_4(current_df):
    lambda_fun = lambda row: row["A+ | No Change"] + row["Q+ | No Change"] + row["W- | No Change"] - row[
        "A+, Q+ | No Change"] \
                             - row["A+, W- | No Change"] - row["Q+, W- | No Change"] + row["A+, Q+, W- | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_5(current_df):
    lambda_fun = lambda row: row["A- | Change"] + row["Q- | Change"] + row["W+ | Change"] - row["A-, Q- | Change"] \
                             - row["A-, W+ | Change"] - row["Q-, W+ | Change"] + row["A-, Q-, W+ | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_no_change_scen_5(current_df):
    lambda_fun = lambda row: row["A- | No Change"] + row["Q- | No Change"] + row["W+ | No Change"] - row[
        "A-, Q- | No Change"] \
                             - row["A-, W+ | No Change"] - row["Q-, W+ | No Change"] + row["A-, Q-, W+ | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_6(current_df):
    lambda_fun = lambda row: row["A- | Change"] + row["Q+ | Change"] + row["W- | Change"] - row["A-, Q+ | Change"] \
                             - row["A-, W- | Change"] - row["Q+, W- | Change"] + row["A-, Q+, W- | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_nochange_scen_6(current_df):
    lambda_fun = lambda row: row["A- | No Change"] + row["Q+ | No Change"] + row["W- | No Change"] - row[
        "A-, Q+ | No Change"] \
                             - row["A-, W- | No Change"] - row["Q+, W- | No Change"] + row["A-, Q+, W- | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_7(current_df):
    lambda_fun = lambda row: row["A+ | Change"] + row["Q- | Change"] + row["W- | Change"] - row["A+, Q- | Change"] \
                             - row["A+, W- | Change"] - row["Q-, W- | Change"] + row["A+, Q-, W- | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_no_change_scen_7(current_df):
    lambda_fun = lambda row: row["A+ | No Change"] + row["Q- | No Change"] + row["W- | No Change"] - row[
        "A+, Q- | No Change"] \
                             - row["A+, W- | No Change"] - row["Q-, W- | No Change"] + row["A+, Q-, W- | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_change_scen_8(current_df):
    lambda_fun = lambda row: row["A- | Change"] + row["Q- | Change"] + row["W- | Change"] - row["A-, Q- | Change"] \
                             - row["A-, W- | Change"] - row["Q-, W- | Change"] + row["A-, Q-, W- | Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    return z


def apply_union_of_events_to_df_given_no_change_scen_8(current_df):
    lambda_fun = lambda row: row["A- | No Change"] + row["Q- | No Change"] + row["W- | No Change"] - row[
        "A-, Q- | No Change"] \
                             - row["A-, W- | No Change"] - row["Q-, W- | No Change"] + row["A-, Q-, W- | No Change"]
    z = [lambda_fun(row) for _, row in current_df.iterrows()]
    assert (np.sum(np.isnan(z)) == 0)
    return z


def assert_fields_of_df_are_not_nan(df):
    assert (df["A+ | Change"].notnull().values.all())
    assert (df["A- | Change"].notnull().values.all())
    assert (df["Q+ | Change"].notnull().values.all())
    assert (df["Q- | Change"].notnull().values.all())
    assert (df["W+ | Change"].notnull().values.all())
    assert (df["W- | Change"].notnull().values.all())
    assert (df["A+, W+ | Change"].notnull().values.all())
    assert (df["A-, W+ | Change"].notnull().values.all())
    assert (df["A+, Q+ | Change"].notnull().values.all())
    assert (df["A-, Q+ | Change"].notnull().values.all())
    assert (df["A+, W- | Change"].notnull().values.all())
    assert (df["A-, W- | Change"].notnull().values.all())
    assert (df["A+, Q- | Change"].notnull().values.all())
    assert (df["A-, Q- | Change"].notnull().values.all())
    assert (df["A+, W- | Change"].notnull().values.all())
    assert (df["A+, W+ | Change"].notnull().values.all())


def return_union_of_probabilities_given_change_to_dataframe(initial_df):
    assert_fields_of_df_are_not_nan(initial_df)
    return_df = pd.DataFrame()
    series_name = "Scen1_Union_change"
    return_df[series_name] = apply_union_of_events_to_df_given_change_scen_1(initial_df)
    return_df["Scen2_Union_change"] = apply_union_of_events_to_df_given_change_scen_2(initial_df)
    return_df["Scen3_Union_change"] = apply_union_of_events_to_df_given_change_scen_3(initial_df)
    return_df["Scen4_Union_change"] = apply_union_of_events_to_df_given_change_scen_4(initial_df)
    return_df["Scen5_Union_change"] = apply_union_of_events_to_df_given_change_scen_5(initial_df)
    return_df["Scen6_Union_change"] = apply_union_of_events_to_df_given_change_scen_6(initial_df)
    return_df["Scen7_Union_change"] = apply_union_of_events_to_df_given_change_scen_7(initial_df)
    return_df["Scen8_Union_change"] = apply_union_of_events_to_df_given_change_scen_8(initial_df)
    #    print(return_df)
    return return_df


def return_union_of_probabilities_given_no_chagne_to_dataframe(initial_df):
    return_df = pd.DataFrame()
    return_df["Scen1_Union_no_change"] = apply_union_of_events_to_df_given_no_change_scen_1(initial_df)
    return_df["Scen2_Union_no_change"] = apply_union_of_events_to_df_given_nochange_scen_2(initial_df)
    return_df["Scen3_Union_no_change"] = apply_union_of_events_to_df_given_no_change_scen_3(initial_df)
    return_df["Scen4_Union_no_change"] = apply_union_of_events_to_df_given_no_change_scen_4(initial_df)
    return_df["Scen5_Union_no_change"] = apply_union_of_events_to_df_given_no_change_scen_5(initial_df)
    return_df["Scen6_Union_no_change"] = apply_union_of_events_to_df_given_nochange_scen_6(initial_df)
    return_df["Scen7_Union_no_change"] = apply_union_of_events_to_df_given_no_change_scen_7(initial_df)
    return_df["Scen8_Union_no_change"] = apply_union_of_events_to_df_given_no_change_scen_8(initial_df)
    return return_df


def return_probability_bounds_to_dataframe_given_change(current_df):
    return_df = pd.DataFrame()
    # Compute s1
    current_df["Scen1_s1"] = current_df[["A+ | Change", "Q+ | Change", "W+ | Change"]].sum(axis=1)
    current_df["Scen2_s1"] = current_df[["A- | Change", "Q+ | Change", "W+ | Change"]].sum(axis=1)
    current_df["Scen3_s1"] = current_df[["A+ | Change", "Q- | Change", "W+ | Change"]].sum(axis=1)
    current_df["Scen4_s1"] = current_df[["A+ | Change", "Q+ | Change", "W- | Change"]].sum(axis=1)
    current_df["Scen5_s1"] = current_df[["A- | Change", "Q- | Change", "W+ | Change"]].sum(axis=1)
    current_df["Scen6_s1"] = current_df[["A- | Change", "Q+ | Change", "W- | Change"]].sum(axis=1)
    current_df["Scen7_s1"] = current_df[["A+ | Change", "Q- | Change", "W- | Change"]].sum(axis=1)
    current_df["Scen8_s1"] = current_df[["A- | Change", "Q- | Change", "W- | Change"]].sum(axis=1)
    local_df = current_df[[
        "Scen1_s1",
        "Scen2_s1",
        "Scen3_s1",
        "Scen4_s1",
        "Scen5_s1",
        "Scen6_s1",
        "Scen7_s1",
        "Scen8_s1"
    ]].copy()
    # Compute s2
    local_df["Scen1_s2"] = current_df[["A+, Q+ | Change", "Q+, W+ | Change", "A+, W+ | Change"]].sum(axis=1)
    local_df["Scen2_s2"] = current_df[["A-, Q+ | Change", "Q+, W+ | Change", "A-, W+ | Change"]].sum(axis=1)
    local_df["Scen3_s2"] = current_df[["A+, Q- | Change", "Q-, W+ | Change", "A+, W+ | Change"]].sum(axis=1)
    local_df["Scen4_s2"] = current_df[["A+, Q+ | Change", "Q+, W- | Change", "A+, W- | Change"]].sum(axis=1)
    local_df["Scen5_s2"] = current_df[["A-, Q- | Change", "Q-, W+ | Change", "A-, W+ | Change"]].sum(axis=1)
    local_df["Scen6_s2"] = current_df[["A-, Q+ | Change", "Q+, W- | Change", "A-, W- | Change"]].sum(axis=1)
    local_df["Scen7_s2"] = current_df[["A+, Q- | Change", "Q-, W- | Change", "A+, W- | Change"]].sum(axis=1)
    local_df["Scen8_s2"] = current_df[["A-, Q- | Change", "Q-, W- | Change", "A-, W- | Change"]].sum(axis=1)
    # Ask for lower and upper bound
    return_df["Scen1_lower_change"] = compute_lower_bound(local_df["Scen1_s1"], local_df["Scen1_s2"])
    return_df["Scen1_upper_change"] = compute_upper_bound(local_df["Scen1_s1"], local_df["Scen1_s2"])
    return_df["Scen2_lower_change"] = compute_lower_bound(local_df["Scen2_s1"], local_df["Scen2_s2"])
    return_df["Scen2_upper_change"] = compute_upper_bound(local_df["Scen2_s1"], local_df["Scen2_s2"])
    return_df["Scen3_lower_change"] = compute_lower_bound(local_df["Scen3_s1"], local_df["Scen3_s2"])
    return_df["Scen3_upper_change"] = compute_upper_bound(local_df["Scen3_s1"], local_df["Scen3_s2"])
    return_df["Scen4_lower_change"] = compute_lower_bound(local_df["Scen4_s1"], local_df["Scen4_s2"])
    return_df["Scen4_upper_change"] = compute_upper_bound(local_df["Scen4_s1"], local_df["Scen4_s2"])
    return_df["Scen5_lower_change"] = compute_lower_bound(local_df["Scen5_s1"], local_df["Scen5_s2"])
    return_df["Scen5_upper_change"] = compute_upper_bound(local_df["Scen5_s1"], local_df["Scen5_s2"])
    return_df["Scen6_lower_change"] = compute_lower_bound(local_df["Scen6_s1"], local_df["Scen6_s2"])
    return_df["Scen6_upper_change"] = compute_upper_bound(local_df["Scen6_s1"], local_df["Scen6_s2"])
    return_df["Scen7_lower_change"] = compute_lower_bound(local_df["Scen7_s1"], local_df["Scen7_s2"])
    return_df["Scen7_upper_change"] = compute_upper_bound(local_df["Scen7_s1"], local_df["Scen7_s2"])
    return_df["Scen8_lower_change"] = compute_lower_bound(local_df["Scen8_s1"], local_df["Scen8_s2"])
    return_df["Scen8_upper_change"] = compute_upper_bound(local_df["Scen8_s1"], local_df["Scen8_s2"])
    return return_df


def return_probability_bounds_to_dataframe_given_no_change(current_df):
    # Compute s1
    current_df["Scen1_s1"] = current_df[["A+ | No Change", "Q+ | No Change", "W+ | No Change"]].sum(axis=1)
    current_df["Scen2_s1"] = current_df[["A- | No Change", "Q+ | No Change", "W+ | No Change"]].sum(axis=1)
    current_df["Scen3_s1"] = current_df[["A+ | No Change", "Q- | No Change", "W+ | No Change"]].sum(axis=1)
    current_df["Scen4_s1"] = current_df[["A+ | No Change", "Q+ | No Change", "W- | No Change"]].sum(axis=1)
    current_df["Scen5_s1"] = current_df[["A- | No Change", "Q- | No Change", "W+ | No Change"]].sum(axis=1)
    current_df["Scen6_s1"] = current_df[["A- | No Change", "Q+ | No Change", "W- | No Change"]].sum(axis=1)
    current_df["Scen7_s1"] = current_df[["A+ | No Change", "Q- | No Change", "W- | No Change"]].sum(axis=1)
    current_df["Scen8_s1"] = current_df[["A- | No Change", "Q- | No Change", "W- | No Change"]].sum(axis=1)
    local_df = current_df[[
        "Scen1_s1",
        "Scen2_s1",
        "Scen3_s1",
        "Scen4_s1",
        "Scen5_s1",
        "Scen6_s1",
        "Scen7_s1",
        "Scen8_s1"
    ]].copy()
    # Compute s2
    local_df["Scen1_s2"] = current_df[["A+, Q+ | No Change", "Q+, W+ | No Change", "A+, W+ | No Change"]].sum(axis=1)
    local_df["Scen2_s2"] = current_df[["A-, Q+ | No Change", "Q+, W+ | No Change", "A-, W+ | No Change"]].sum(axis=1)
    local_df["Scen3_s2"] = current_df[["A+, Q- | No Change", "Q-, W+ | No Change", "A+, W+ | No Change"]].sum(axis=1)
    local_df["Scen4_s2"] = current_df[["A+, Q+ | No Change", "Q+, W- | No Change", "A+, W+ | No Change"]].sum(axis=1)
    local_df["Scen5_s2"] = current_df[["A-, Q- | No Change", "Q-, W+ | No Change", "A-, W+ | No Change"]].sum(axis=1)
    local_df["Scen6_s2"] = current_df[["A-, Q+ | No Change", "Q+, W- | No Change", "A-, W- | No Change"]].sum(axis=1)
    local_df["Scen7_s2"] = current_df[["A+, Q- | No Change", "Q-, W- | No Change", "A+, W- | No Change"]].sum(axis=1)
    local_df["Scen8_s2"] = current_df[["A-, Q- | No Change", "Q-, W- | No Change", "A-, W- | No Change"]].sum(axis=1)
    # Ask for lower and upper bound
    return_df = pd.DataFrame()
    return_df["Scen1_lower_no_change"] = compute_lower_bound(local_df["Scen1_s1"], local_df["Scen1_s2"])
    return_df["Scen1_upper_no_change"] = compute_upper_bound(local_df["Scen1_s1"], local_df["Scen1_s2"])
    return_df["Scen2_lower_no_change"] = compute_lower_bound(local_df["Scen2_s1"], local_df["Scen2_s2"])
    return_df["Scen2_upper_no_change"] = compute_upper_bound(local_df["Scen2_s1"], local_df["Scen2_s2"])
    return_df["Scen3_lower_no_change"] = compute_lower_bound(local_df["Scen3_s1"], local_df["Scen3_s2"])
    return_df["Scen3_upper_no_change"] = compute_upper_bound(local_df["Scen3_s1"], local_df["Scen3_s2"])
    return_df["Scen4_lower_no_change"] = compute_lower_bound(local_df["Scen4_s1"], local_df["Scen4_s2"])
    return_df["Scen4_upper_no_change"] = compute_upper_bound(local_df["Scen4_s1"], local_df["Scen4_s2"])
    return_df["Scen5_lower_no_change"] = compute_lower_bound(local_df["Scen5_s1"], local_df["Scen5_s2"])
    return_df["Scen5_upper_no_change"] = compute_upper_bound(local_df["Scen5_s1"], local_df["Scen5_s2"])
    return_df["Scen6_lower_no_change"] = compute_lower_bound(local_df["Scen6_s1"], local_df["Scen6_s2"])
    return_df["Scen6_upper_no_change"] = compute_upper_bound(local_df["Scen6_s1"], local_df["Scen6_s2"])
    return_df["Scen7_lower_no_change"] = compute_lower_bound(local_df["Scen7_s1"], local_df["Scen7_s2"])
    return_df["Scen7_upper_no_change"] = compute_upper_bound(local_df["Scen7_s1"], local_df["Scen7_s2"])
    return_df["Scen8_lower_no_change"] = compute_lower_bound(local_df["Scen8_s1"], local_df["Scen8_s2"])
    return_df["Scen8_upper_no_change"] = compute_upper_bound(local_df["Scen8_s1"], local_df["Scen8_s2"])
    return return_df


def return_valid_cases(main_df):
    # for each scenario returns true if the all probability is true and false otherwise
    cond_1 = main_df["Scen1_Union_change"].ge(main_df["Scen1_lower_change"])
    cond_2 = main_df["Scen1_Union_change"].le(main_df["Scen1_upper_change"])
    main_df["Scen1_change_valid"] = cond_1 & cond_2
    main_df["Scen1_no_change_valid"] = (main_df["Scen1_lower_no_change"].le(main_df["Scen1_Union_no_change"])
                                        & main_df["Scen1_Union_no_change"].le(main_df["Scen1_upper_no_change"]))
    main_df["Scen2_change_valid"] = (main_df["Scen2_lower_change"].le(main_df["Scen2_Union_change"])
                                     & main_df["Scen2_Union_change"].le(main_df["Scen2_upper_change"]))
    main_df["Scen2_no_change_valid"] = (main_df["Scen2_lower_no_change"].le(main_df["Scen2_Union_no_change"])
                                        & main_df["Scen2_Union_no_change"].le(main_df["Scen2_upper_no_change"]))
    main_df["Scen3_change_valid"] = (main_df["Scen3_lower_change"].le(main_df["Scen3_Union_change"]) &
                                     main_df["Scen3_Union_change"].le(main_df["Scen3_upper_change"]))
    main_df["Scen3_no_change_valid"] = (main_df["Scen3_lower_no_change"].le(main_df["Scen3_Union_no_change"])
                                        & main_df["Scen3_Union_no_change"].le(main_df["Scen3_upper_no_change"]))
    main_df["Scen4_change_valid"] = (main_df["Scen4_lower_change"].le(main_df["Scen4_Union_change"])
                                     & main_df["Scen4_Union_change"].le(main_df["Scen4_upper_change"]))
    main_df["Scen4_no_change_valid"] = (main_df["Scen4_lower_no_change"].le(main_df["Scen4_Union_no_change"])
                                        & main_df["Scen4_Union_no_change"].le(main_df["Scen4_upper_no_change"]))
    main_df["Scen5_change_valid"] = (main_df["Scen5_lower_change"].le(main_df["Scen5_Union_change"])
                                     & main_df["Scen5_Union_change"].le(main_df["Scen5_upper_change"]))
    main_df["Scen5_no_change_valid"] = (main_df["Scen5_lower_no_change"].le(main_df["Scen5_Union_no_change"])
                                        & main_df["Scen5_Union_no_change"].le(main_df["Scen5_upper_no_change"]))
    main_df["Scen6_change_valid"] = (main_df["Scen6_lower_change"].le(main_df["Scen6_Union_change"])
                                     & main_df["Scen6_Union_change"].le(main_df["Scen6_upper_change"]))
    main_df["Scen6_no_change_valid"] = (main_df["Scen6_lower_no_change"].le(main_df["Scen6_Union_no_change"])
                                        & main_df["Scen6_Union_no_change"].le(main_df["Scen6_upper_no_change"]))
    main_df["Scen7_change_valid"] = (main_df["Scen7_lower_change"].le(main_df["Scen7_Union_change"])
                                     & main_df["Scen7_Union_change"].le(main_df["Scen7_upper_change"]))
    main_df["Scen7_no_change_valid"] = (main_df["Scen7_lower_no_change"].le(main_df["Scen7_Union_no_change"])
                                        & main_df["Scen7_Union_no_change"].le(main_df["Scen7_upper_no_change"]))
    main_df["Scen8_change_valid"] = (main_df["Scen8_lower_change"].le(main_df["Scen8_Union_change"])
                                     & main_df["Scen8_Union_change"].le(main_df["Scen8_upper_change"]))
    main_df["Scen8_no_change_valid"] = (main_df["Scen8_lower_no_change"].le(main_df["Scen8_Union_no_change"])
                                        & main_df["Scen8_Union_no_change"].le(main_df["Scen8_upper_no_change"]))
    return main_df


def return_dictionary_of_valid_bounds(main_df):
    num_valid_bound_dic = {}
    n = len(main_df)
    for scenario_num in range(1, 9):
        num_valid_bound_dic[f"Scen{scenario_num}_change"] = main_df[f"Scen{scenario_num}_change_valid"].sum()
        num_valid_bound_dic[f"Scen{scenario_num}_change_invalid"] = n - main_df[
            f"Scen{scenario_num}_change_valid"].sum()
        num_valid_bound_dic[f"Scen{scenario_num}_no_change"] = main_df[f"Scen{scenario_num}_no_change_valid"].sum()
        num_valid_bound_dic[f"Scen{scenario_num}_no_change_invalid"] = n - main_df[
            f"Scen{scenario_num}_no_change_valid"].sum()
    return num_valid_bound_dic


def save_results_dic_to_csv(results_dic):
    csv_file_name = "Results/GLRT_ROSS/Performance_Tests/Bonferroni_Num_Valid_Bounds.csv"
    with open(csv_file_name, 'w') as f:
        for key, value in results_dic.items():
            f.write("%s, %s\n" % (key, value))


def clean_orig_df(orig_df):
    orig_df = orig_df.apply(pd.to_numeric)
    return orig_df


def plot_colormap_of_probability_results(a_df):
    # that's where it gets complicated.
    # I want to plot all the values not the bound
    # and then use the bounds in the last column to  decide which square will be colored black
    # solution.
    # Copy the dataframe only the actual values of the probabilities union
    # mask or set the out of bound values to NaN or to a very high negative value that would be out of bound
    # Then plot the heatmatp
    output_df = a_df.copy()
    scenario_prob_columns = [
        "Scen1_Union_change",
        "Scen2_Union_change",
        "Scen3_Union_change",
        "Scen4_Union_change",
        "Scen5_Union_change",
        "Scen6_Union_change",
        "Scen7_Union_change",
        "Scen8_Union_change",
        "Scen1_Union_no_change",
        "Scen2_Union_no_change",
        "Scen3_Union_no_change",
        "Scen4_Union_no_change",
        "Scen5_Union_no_change",
        "Scen6_Union_no_change",
        "Scen7_Union_no_change",
        "Scen8_Union_no_change"
    ]
    for prob_str in scenario_prob_columns:
        # 1. find the number in the string
        num_found = search('Scen([0-9]*)_Union', prob_str).group(1)
        prefix = "Scen"
        if "no_change" in prob_str:
            upper_suffix = "_upper_no_change"
            lower_suffix = "_lower_no_change"
            upper_bound_column_str = ''.join([prefix, num_found, upper_suffix])
            lower_bound_column_str = ''.join([prefix, num_found, lower_suffix])
        else:
            upper_suffix = "_upper_change"
            lower_suffix = "_lower_change"
            upper_bound_column_str = ''.join([prefix, num_found, upper_suffix])
            lower_bound_column_str = ''.join([prefix, num_found, lower_suffix])
        for rho in output_df["rho"].unique():
            rho_output_df = output_df[output_df["rho"] == rho]
            prob_series = rho_output_df[prob_str]
            max_prob_value = prob_series.max()
            min_prob_value = prob_series.min()
            print(
                f"For rho={rho} and scenario: {prob_str},\n\t The maximum value is {max_prob_value} and the minimum is {min_prob_value}.")
            rho_output_df[rho_output_df[prob_str] > rho_output_df[upper_bound_column_str]][prob_str] = \
                max_prob_value + 10
            rho_output_df[rho_output_df[prob_str] < rho_output_df[lower_bound_column_str]][prob_str] = \
                min_prob_value - 10
            df_for_heatmap = pd.pivot_table(
                data=rho_output_df,
                index="delta_rho",
                columns="Batch Size",
                values=prob_str
            )
            df_for_heatmap.head()
            local_heat_map = sns.heatmap(df_for_heatmap, vmin=min_prob_value, vmax=max_prob_value)
            fig = local_heat_map.get_figure()
            fig.savefig(f"Figures/Bernoulli_Validation/heatmap_{prob_str}_rho_{int(rho * 100)}.png")
            plt.close(fig)


if __name__ == "__main__":
    # do something
    joint_prob_df = return_joint_probabilities_df_given_change()
    assert_fields_of_df_are_not_nan(joint_prob_df)
    joint_prob_df = clean_orig_df(joint_prob_df)
    # print(joint_prob_df.columns)
    print(joint_prob_df.info())

    initial_df = joint_prob_df[["rho", "delta_rho", "Batch Size"]]
    union_of_prob_given_change_df = return_union_of_probabilities_given_change_to_dataframe(joint_prob_df)
    union_of_prob_given_no_change_df = return_union_of_probabilities_given_no_chagne_to_dataframe(joint_prob_df)
    #    print(union_of_prob_given_no_change_df)
    prob_bounds_df_no_change = return_probability_bounds_to_dataframe_given_no_change(joint_prob_df)
    #    print(prob_bounds_df_no_change)
    prob_bounds_df_change = return_probability_bounds_to_dataframe_given_change(joint_prob_df)
    #    print(prob_bounds_df_change)
    all_probs_df = pd.concat([initial_df, union_of_prob_given_change_df, union_of_prob_given_no_change_df,
                              prob_bounds_df_change, prob_bounds_df_no_change], axis=1)
    #    # make the checks for each scenario
    print("\n*Produced new df*")
    print(all_probs_df.info())
    print(all_probs_df.head())
    all_probs_df.to_csv("Results/GLRT_ROSS/Performance_Tests/validation_of_Bernoulli_inequalities.csv")
    all_probs_df = return_valid_cases(all_probs_df)
    # return a dictionary with hte number of valid and invalid bounds
    num_correct_bounds_dic = return_dictionary_of_valid_bounds(all_probs_df)
    print(num_correct_bounds_dic)
    save_results_dic_to_csv(num_correct_bounds_dic)
    plot_colormap_of_probability_results(all_probs_df)
