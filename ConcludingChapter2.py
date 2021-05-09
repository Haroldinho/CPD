"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: ConcludingChapter2.py
Description: Chapter 2 of my thesis is about single change point detection for a M/M/1 queueing system with
a single type of observation either waiting times, avg time spent in queue (age) or length of the queue (L_Q)
This doc serves as a way to collect the output statistics in the file:
"joint_conditional_probability_resultsofApril2020.csv" to determine the best batch size for simulation conition.
A simulation is defined by the initial traffic intensity (rho)and the relative traffic intensity change (deltarho)
Author: Harold Nemo Adodo Nikoue
part of my partial observability thesis
Date: 5/02/2021
"""

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Steps
# Step 1: Load conditional probability of all combinations of outcome given a changepoint or no changepoint

main_file_name = "./Results/GLRT_ROSS/Performance_Tests/joint_conditional_probability.csv"
joint_cond_prob_df = pd.read_csv(main_file_name)

# Step 2: Reduce the dataframe only to single conditional outcomes
# A+ | Change, A- | Change, W+ | Change ,....
cond_prob_df = joint_cond_prob_df[[
    "rho", "delta_rho", "Batch Size",
    "A+ | Change",
    "A- | Change",
    "W+ | Change",
    "W- | Change",
    "Q+ | Change",
    "Q- | Change",
    "A+ | No Change",
    "A- | No Change",
    "W+ | No Change",
    "W- | No Change",
    "Q+ | No Change",
    "Q- | No Change"
]]
# Assert that all the probabilities make change
# Should probably be in a test file


# Step 3:   Rename X+ | No Change to ErrorTypeI_X
#           Rename X- | Change to ErrorTypeII_X

cond_prob_df = cond_prob_df.rename(columns=
{
    "A- | Change": "ErrorTypeI_A",
    "W- | Change": "ErrorTypeI_W",
    "Q- | Change": "ErrorTypeI_Q",
    "A+ | No Change": "ErrorTypeII_A",
    "Q+ | No Change": "ErrorTypeII_Q",
    "W+ | No Change": "ErrorTypeII_W",
})
print(cond_prob_df.columns)

####################################################
# Read Detection delay files by type
##################################################

#  Age
# Age_of_Process/SimpleDetection_age_of_process_Batch_of_size_1000_rcpm.csv
age_folder_name = "./Results/GLRT_ROSS/Performance_Tests/Age_of_Process/"
all_age_files = [f for f in listdir(age_folder_name) if isfile(join(age_folder_name, f))]
full_df_list = []
for file_name in all_age_files:
    if file_name.endswith(".csv"):
        full_file_name = age_folder_name + file_name
        local_df = pd.read_csv(full_file_name)
        full_df_list.append(local_df)
full_age_df = pd.concat(full_df_list, axis=0).rename(columns={"batch size": "Batch Size", "arl_1": "ARL_1"})

#  Queue Length
# Queue_Length/SimpleDetection_queue_length_Batch_of_size_100_rcpm.csv
queue_folder_name = "./Results/GLRT_ROSS/Performance_Tests/Queue_Length/"
all_queue_files = [f for f in listdir(queue_folder_name) if isfile(join(queue_folder_name, f))]
full_df_list = []
for file_name in all_queue_files:
    if file_name.endswith(".csv"):
        local_df = pd.read_csv(queue_folder_name + file_name)
        full_df_list.append(local_df)
full_queue_df = pd.concat(full_df_list, axis=0).rename(columns={"batch size": "Batch Size", "arl_1": "ARL_1"})

# Wait_Times/SimpleDetection_Batch_of_size_1000_rcpm.csv
wait_folder_name = "./Results/GLRT_ROSS/Performance_Tests/Wait_Times/"
all_wait_files = [f for f in listdir(wait_folder_name) if isfile(join(wait_folder_name, f))]
full_df_list = []
for file_name in all_wait_files:
    if file_name.endswith(".csv"):
        local_df = pd.read_csv(wait_folder_name + file_name)
        full_df_list.append(local_df)
full_wait_df = pd.concat(full_df_list, axis=0).rename(columns={"batch size": "Batch Size", "arl_1": "ARL_1"})
print("Columns of full_wait_df: ", full_wait_df.columns)

# Iterate over all rhos and delta rhos
list_of_rhos = list(cond_prob_df["rho"].unique())
list_of_delta_rhos = list(cond_prob_df["delta_rho"].unique())
idx = 0
list_of_parameters = []
selected_combinations_A = []
best_batch_size_option1_A = []
best_batch_size_option2_A = []
best_batch_size_option3_A = []
lowest_false_negatives_A = []
lowest_ARL_1_A = []
lowest_ARL_1_W = []
lowest_ARL_1_Q = []

selected_combinations_Q = []
best_batch_size_option1_Q = []
best_batch_size_option2_Q = []
best_batch_size_option3_Q = []
lowest_false_negatives_Q = []
selected_combinations_W = []
best_batch_size_option1_W = []
best_batch_size_option2_W = []
lowest_false_negatives_W = []
best_batch_size_option3_W = []

z_dic_of_dic_batch_option_1_A = {}
z_dic_of_dic_batch_option_2_A = {}
z_dic_of_dic_batch_option_3_A = {}
z_dic_of_dic_batch_option_1_Q = {}
z_dic_of_dic_batch_option_2_Q = {}
z_dic_of_dic_batch_option_3_Q = {}
z_dic_of_dic_batch_option_1_W = {}
z_dic_of_dic_batch_option_2_W = {}
z_dic_of_dic_batch_option_3_W = {}
for rho in list_of_rhos:
    z_dic_of_dic_batch_option_3_A[rho] = {}
    z_dic_of_dic_batch_option_2_A[rho] = {}
    z_dic_of_dic_batch_option_1_A[rho] = {}

    z_dic_of_dic_batch_option_3_Q[rho] = {}
    z_dic_of_dic_batch_option_2_Q[rho] = {}
    z_dic_of_dic_batch_option_1_Q[rho] = {}

    z_dic_of_dic_batch_option_3_W[rho] = {}
    z_dic_of_dic_batch_option_2_W[rho] = {}
    z_dic_of_dic_batch_option_1_W[rho] = {}
    for delta_rho in list_of_delta_rhos:
        print(f"Current rho {rho}, current delta rho {delta_rho}")
        list_of_parameters.append((rho, delta_rho))
        idx += 1
        local_df = cond_prob_df[(cond_prob_df["rho"] == rho) & (cond_prob_df["delta_rho"] == delta_rho)]
        selected_df_A = local_df[local_df["ErrorTypeI_A"] <= 0.05]
        selected_df_Q = local_df[local_df["ErrorTypeI_Q"] <= 0.05]
        selected_df_W = local_df[local_df["ErrorTypeI_W"] <= 0.05]
        if len(selected_df_A) == 0:
            print(
                f"For rho={rho} and delta_rho={delta_rho}, there is no test that satisfies the FAR requirement for the age of a process.")
        else:
            selected_combinations_A.append((rho, delta_rho))
            # there are some rows that satisfies the first requirement
            # Step 4: Limit the df to all rows for which ErrorTypeI_X <= 0.05
            ########################################################################################
            #   Option 1: Select the lowest batch size that satisfies the maximum FAR constraint
            # Step 5: Pick the lowest batch size that fulfills that requirement and store it in a dictionary
            #########################################################################################
            best_batch_size_option1_A.append(min(selected_df_A["Batch Size"]))
            z_dic_of_dic_batch_option_1_A[rho][delta_rho] = best_batch_size_option1_A[-1]

            #######################################################################################################
            #   Option 2: Select the batch size that satisfies the FAR constraint and minimizes the Type II error
            # Step 6: In the new dataframe, pick the row with the lowest ErrorTypeII_X
            ##########################################################################################################

            lowest_no_detection_prob = min(selected_df_A["ErrorTypeI_A"])
            selected_rows = selected_df_A[selected_df_A["ErrorTypeI_A"] == lowest_no_detection_prob][
                "Batch Size"].to_list()
            best_batch_size_option2_A.append(selected_rows[0])
            z_dic_of_dic_batch_option_2_A[rho][delta_rho] = best_batch_size_option2_A[-1]
            lowest_false_negatives_A.append(lowest_no_detection_prob)
            # Step 7: Merge the current df with the data with detection delays data
            # Merge on batch size, rho, delta rho,
            joined_df = pd.merge(selected_df_A, full_age_df, on=["rho", "delta_rho", "Batch Size"], how="inner")
            # pick the batch size with the lowest ARL_1
            if len(joined_df) == 0:
                best_batch_size_option3_A.append(np.nan)
                lowest_ARL_1_A.append(np.nan)
            else:
                min_ARL_1 = min(joined_df["ARL_1"])
                if np.isnan(min_ARL_1):
                    best_batch_size_option3_A.append(np.nan)
                else:
                    selected_row = joined_df[joined_df["ARL_1"] == min_ARL_1]["Batch Size"].to_list()
                    best_batch_size_option3_A.append(selected_row[0])
                lowest_ARL_1_A.append(min_ARL_1)
            z_dic_of_dic_batch_option_3_A[rho][delta_rho] = best_batch_size_option3_A[-1]
        # Queue Length
        if len(selected_df_Q) == 0:
            print(f"For rho={rho} and delta_rho={delta_rho}, there is no test that satisfies the FAR requirement [Q].")
        else:
            selected_combinations_Q.append((rho, delta_rho))
            best_batch_size_option1_Q.append(min(selected_df_Q["Batch Size"]))
            z_dic_of_dic_batch_option_1_Q[rho][delta_rho] = best_batch_size_option1_Q[-1]

            lowest_no_detection_prob = min(selected_df_Q["ErrorTypeI_Q"])
            selected_rows = selected_df_Q[selected_df_Q["ErrorTypeI_Q"] == lowest_no_detection_prob][
                "Batch Size"].to_list()
            best_batch_size_option2_Q.append(selected_rows[0])
            z_dic_of_dic_batch_option_2_Q[rho][delta_rho] = best_batch_size_option2_Q[-1]
            lowest_false_negatives_Q.append(lowest_no_detection_prob)
            joined_df = pd.merge(selected_df_Q, full_queue_df, on=["rho", "delta_rho", "Batch Size"], how="inner")
            # pick the batch size with the lowest ARL_1
            if len(joined_df) == 0:
                best_batch_size_option3_Q.append(np.nan)
                lowest_ARL_1_Q.append(np.nan)
            else:
                min_ARL_1 = min(joined_df["ARL_1"])
                if np.isnan(min_ARL_1):
                    best_batch_size_option3_Q.append(np.nan)
                else:
                    selected_row = joined_df[joined_df["ARL_1"] == min_ARL_1]["Batch Size"].to_list()
                    best_batch_size_option3_Q.append(selected_row[0])
                lowest_ARL_1_Q.append(min_ARL_1)
            z_dic_of_dic_batch_option_3_Q[rho][delta_rho] = best_batch_size_option3_Q[-1]
        # Waiting Times
        if len(selected_df_W) == 0:
            print(f"For rho={rho} and delta_rho={delta_rho}, there is no test that satisfies the FAR requirement [W].")
        else:
            selected_combinations_W.append((rho, delta_rho))
            best_batch_size_option1_W.append(min(selected_df_W["Batch Size"]))
            z_dic_of_dic_batch_option_1_W[rho][delta_rho] = best_batch_size_option1_W[-1]

            lowest_no_detection_prob = min(selected_df_W["ErrorTypeI_W"])
            lowest_false_negatives_W.append(lowest_no_detection_prob)
            selected_rows = selected_df_W[selected_df_W["ErrorTypeI_W"] ==
                                          lowest_no_detection_prob]["Batch Size"].to_list()
            best_batch_size_option2_W.append(selected_rows[0])
            z_dic_of_dic_batch_option_2_W[rho][delta_rho] = best_batch_size_option2_W[-1]
            joined_df = pd.merge(selected_df_W, full_wait_df, on=["rho", "delta_rho", "Batch Size"], how="inner")
            # pick the batch size with the lowest ARL_1
            if len(joined_df) < 1:
                best_batch_size_option3_W.append(np.nan)
                lowest_ARL_1_W.append(np.nan)
            else:
                min_ARL_1 = min(joined_df["ARL_1"])
                if np.isnan(min_ARL_1):
                    best_batch_size_option3_W.append(np.nan)
                else:
                    selected_row = joined_df[joined_df["ARL_1"] == min_ARL_1]["Batch Size"].to_list()
                    best_batch_size_option3_W.append(selected_row[0])
                lowest_ARL_1_W.append(min_ARL_1)
            z_dic_of_dic_batch_option_3_W[rho][delta_rho] = best_batch_size_option3_W[-1]
print("------------------------------ Age of Process --------------------------------")
print("All parameters:", list_of_parameters)
print("Parameters with a combination: ", selected_combinations_A)
print("Option 1:")
print("\t batch sizes: ", best_batch_size_option1_A)
print("\nOption 2:")
print("\t batch sizes: ", best_batch_size_option2_A)
print("\t false negative rates: ", lowest_false_negatives_A)
combination_not_represented = set(list_of_parameters) - set(selected_combinations_A)
print("Combinations not represented: ", combination_not_represented)
print("------------------------------------------------------------------------------")

print("------------------------------ Waiting Times --------------------------------")
print("Wll parameters:", list_of_parameters)
print("Parameters with a combination: ", selected_combinations_W)
print("Option 1:")
print("\t batch sizes: ", best_batch_size_option1_W)
print("\nOption 2:")
print("\t batch sizes: ", best_batch_size_option2_W)
print("\t false negative rates: ", lowest_false_negatives_W)
combination_not_represented = set(list_of_parameters) - set(selected_combinations_W)
print("Combinations not represented: ", combination_not_represented)
print("------------------------------------------------------------------------------")

print("------------------------------ Queue length --------------------------------")
print("All parameters:", list_of_parameters)
print("Parameters with a combination: ", selected_combinations_Q)
print("Option 1:")
print("\t batch sizes: ", best_batch_size_option1_Q)
print("\nOption 2:")
print("\t batch sizes: ", best_batch_size_option2_Q)
print("\t false negative rates: ", lowest_false_negatives_Q)
combination_not_represented = set(list_of_parameters) - set(selected_combinations_Q)
print("Combinations not represented: ", combination_not_represented)
print("------------------------------------------------------------------------------")
# Step 7: Merge the current df with the data with detection delays data
# Queue_Length/SimpleDetection_queue_length_Batch_of_size_100_rcpm.csv
# Wait_Times/SimpleDetection_Batch_of_size_1000_rcpm.csv
# Step 8: Pick the rows that have a batch size at least as big as the one in the primary df
# and pick the batch size that gives the lowest detection delay amongst the results.

# Create a dataframe of the results for different rho and delta rhos
# For age: A
x_list_of_rho_A = list(set([age_idx[0] for age_idx in selected_combinations_A]))
y_list_of_delta_rho_A = list(set([age_idx[1] for age_idx in selected_combinations_A]))

# convert the dataframe into a matrix
z_list_of_list_batch_option_1_A = []
z_list_of_list_batch_option_2_A = []
z_list_of_list_batch_option_3_A = []
for delta_rho in y_list_of_delta_rho_A:
    z_inner_list_1 = []
    z_inner_list_2 = []
    z_inner_list_3 = []
    for rho in x_list_of_rho_A:
        if delta_rho in z_dic_of_dic_batch_option_1_A[rho]:
            z_inner_list_1.append(z_dic_of_dic_batch_option_1_A[rho][delta_rho])
            z_inner_list_2.append(z_dic_of_dic_batch_option_2_A[rho][delta_rho])
            z_inner_list_3.append(z_dic_of_dic_batch_option_3_A[rho][delta_rho])
        else:
            z_inner_list_1.append(np.nan)
            z_inner_list_2.append(np.nan)
            z_inner_list_3.append(np.nan)
    z_list_of_list_batch_option_1_A.append(z_inner_list_1)
    z_list_of_list_batch_option_2_A.append(z_inner_list_2)
    z_list_of_list_batch_option_3_A.append(z_inner_list_3)

# For Queue
x_list_of_rho_Q = list(set([age_idx[0] for age_idx in selected_combinations_Q]))
y_list_of_delta_rho_Q = list(set([age_idx[1] for age_idx in selected_combinations_Q]))

# convert the dataframe into a matrix
z_list_of_list_batch_option_1_Q = []
z_list_of_list_batch_option_2_Q = []
z_list_of_list_batch_option_3_Q = []
for delta_rho in y_list_of_delta_rho_Q:
    z_inner_list_1 = []
    z_inner_list_2 = []
    z_inner_list_3 = []
    for rho in x_list_of_rho_Q:
        if delta_rho in z_dic_of_dic_batch_option_1_Q[rho]:
            z_inner_list_1.append(z_dic_of_dic_batch_option_1_Q[rho][delta_rho])
            z_inner_list_2.append(z_dic_of_dic_batch_option_2_Q[rho][delta_rho])
            z_inner_list_3.append(z_dic_of_dic_batch_option_3_Q[rho][delta_rho])
        else:
            z_inner_list_1.append(np.nan)
            z_inner_list_2.append(np.nan)
            z_inner_list_3.append(np.nan)
    z_list_of_list_batch_option_1_Q.append(z_inner_list_1)
    z_list_of_list_batch_option_2_Q.append(z_inner_list_2)
    z_list_of_list_batch_option_3_Q.append(z_inner_list_3)

# For Waiting-Times
x_list_of_rho_W = list(set([age_idx[0] for age_idx in selected_combinations_W]))
y_list_of_delta_rho_W = list(set([age_idx[1] for age_idx in selected_combinations_W]))

# convert the dataframe into a matrix
z_list_of_list_batch_option_1_W = []
z_list_of_list_batch_option_2_W = []
z_list_of_list_batch_option_3_W = []
for delta_rho in y_list_of_delta_rho_W:
    z_inner_list_1 = []
    z_inner_list_2 = []
    z_inner_list_3 = []
    for rho in x_list_of_rho_W:
        if delta_rho in z_dic_of_dic_batch_option_1_W[rho]:
            z_inner_list_1.append(z_dic_of_dic_batch_option_1_W[rho][delta_rho])
            z_inner_list_2.append(z_dic_of_dic_batch_option_2_W[rho][delta_rho])
            z_inner_list_3.append(z_dic_of_dic_batch_option_3_W[rho][delta_rho])
        else:
            z_inner_list_1.append(np.nan)
            z_inner_list_2.append(np.nan)
            z_inner_list_3.append(np.nan)
    z_list_of_list_batch_option_1_W.append(z_inner_list_1)
    z_list_of_list_batch_option_2_W.append(z_inner_list_2)
    z_list_of_list_batch_option_3_W.append(z_inner_list_3)


def heatmap_for_batches(z_mat, x_list, y_list, plot_name):
    # Heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(z_mat)
    # Show all ticks
    ax.set_xticks(np.arange(len(x_list)))
    ax.set_yticks(np.arange(len(y_list)))
    # label them with the respective list entries
    ax.set_xticklabels(x_list)
    ax.set_yticklabels(y_list)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbarlabel = "Batch Size"
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel("rho")
    plt.ylabel("delta rho")
    fig.tight_layout()
    plt.savefig(plot_name, dpi=500)
    plt.close()


heatmap_for_batches(z_list_of_list_batch_option_2_A, x_list_of_rho_A, y_list_of_delta_rho_A,
                    "Results/Chapter2ContourPlots/TypeIIMinBatchSizeAgeProcess.png")
heatmap_for_batches(z_list_of_list_batch_option_1_A, x_list_of_rho_A, y_list_of_delta_rho_A,
                    "Results/Chapter2ContourPlots/TypeIMinBatchSizeAgeProcess.png")
heatmap_for_batches(z_list_of_list_batch_option_3_A, x_list_of_rho_A, y_list_of_delta_rho_A,
                    "Results/Chapter2ContourPlots/DetectionDelayMinBatchSizeAgeProcess.png")
heatmap_for_batches(z_list_of_list_batch_option_2_Q, x_list_of_rho_Q, y_list_of_delta_rho_Q,
                    "Results/Chapter2ContourPlots/TypeIIMinBatchSizeQueueLength.png")
heatmap_for_batches(z_list_of_list_batch_option_1_Q, x_list_of_rho_Q, y_list_of_delta_rho_Q,
                    "Results/Chapter2ContourPlots/TypeIMinBatchSizeQueueLength.png")
heatmap_for_batches(z_list_of_list_batch_option_3_Q, x_list_of_rho_Q, y_list_of_delta_rho_Q,
                    "Results/Chapter2ContourPlots/DetectionDelayMinBatchSizeQueueLength.png")
heatmap_for_batches(z_list_of_list_batch_option_2_W, x_list_of_rho_W, y_list_of_delta_rho_W,
                    "Results/Chapter2ContourPlots/TypeIIMinBatchSizeWaitTime.png")
heatmap_for_batches(z_list_of_list_batch_option_1_W, x_list_of_rho_W, y_list_of_delta_rho_W,
                    "Results/Chapter2ContourPlots/TypeIMinBatchSizeWaitTime.png")
heatmap_for_batches(z_list_of_list_batch_option_3_W, x_list_of_rho_W, y_list_of_delta_rho_W,
                    "Results/Chapter2ContourPlots/DetectionDelayMinBatchSizeWaitTime.png")
