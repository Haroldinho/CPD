"""
Plot results from the pkl files

Step 1: load the pkl files

Step 2: Extract the data from these files

Step 3: plot the results
"""
import seaborn as sns
from utilities import PowerTestLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pkl_data():
    # 1. Load the pkl files
    pkl_directory = "./Results/GLRT_ROSS/"
    pkl_file_1 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_1_05_22_2020.pkl"
    pkl_file_2 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_25_05_21_2020.pkl"
    pkl_file_3 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_50_05_21_2020.pkl"
    pkl_file_4 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_80_05_23_2020.pkl"
    pkl_file_5 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_100_05_23_2020.pkl"
    pkl_file_6 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_150_05_20_2020.pkl"
    pkl_file_7 = "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_200_05_21_2020.pkl"
    print(eval("pkl_file_1"))
    pkl_files_string = "["
    for i in range(1, 7):
        pkl_files_string += "pkl_file_{:d},".format(i)
    pkl_files_string += "pkl_file_7]"
    print(pkl_files_string)
    pkl_files = eval(pkl_files_string)
    # pkl_files = [pkl_file_1, pkl_file_3, pkl_file_4, pkl_file_5, pkl_file_6, pkl_file_7, pkl_file_8, pkl_file_9,
    #             pkl_file_10]
    batch_size = [1, 25, 50, 80, 100, 150, 200]
    # default_detection_thresholds = np.linspace(.1, 20, 20)
    test_dic = {b: [] for b in batch_size}
    # Create a long form dataframe
    long_form_dataframe = pd.DataFrame(columns=["Batch Size", "Threshold", "ARL_0"])
    index = 0
    for idx, file_name in enumerate(pkl_files):
        pkl_logger = PowerTestLogger(pkl_directory + file_name, is_full_path=True)
        batch = batch_size[idx]
        pkl_data = pkl_logger.load_data()
        detection_thresholds_lists = list(pkl_data.values())
        arl_0_list = list(pkl_data.keys())
        # I am creating a dictionary
        threshold_arl_dict = {}
        if isinstance(detection_thresholds_lists[0], (list, tuple, np.ndarray)):
            detection_thresholds = detection_thresholds_lists[0]
            # print(detection_thresholds)
            for iddx in range(len(arl_0_list)):
                # if iddx < len(arl_0_list):
                threshold = detection_thresholds[iddx]
                threshold_arl_dict[threshold] = arl_0_list[iddx]
                values_to_add = [batch, threshold, arl_0_list[iddx]]
                long_form_dataframe.loc[index] = values_to_add
                index += 1
            # print(long_form_dataframe)
        else:
            for arl_0, threshold in pkl_data.items():
                threshold_arl_dict[threshold] = arl_0
                values_to_add = [batch, threshold, arl_0]
                long_form_dataframe.loc[index] = values_to_add
                index += 1
        test_dic[batch] = threshold_arl_dict
    return long_form_dataframe


def plot_violin_data_type_1(df):
    sns.set(style="whitegrid")
    ax = sns.violinplot(x="Batch Size", y="Threshold", data=df, palette="muted")
    plt.title('Threshold vs. Batch Size')
    plt.savefig("./Figures/threshold_vs_batch_violin.png", dpi=500)
    plt.show()
    plt.close()


def plot_violin_data_type_2(df):
    sns.set(style="whitegrid")
    ax = sns.violinplot(x="Threshold", y="ARL_0", hue="Batch Size", data=df, palette="muted")
    plt.title('ARL vs. Threshold')
    plt.savefig("./Figures/arl_0_vs_threshold_violin.png", dpi=500)
    plt.show()
    plt.close()


def plot_violin_data_type_3(df):
    sns.set(style="whitegrid")
    ax = sns.violinplot(x="Batch Size", y="ARL_0", data=df, palette="muted")
    plt.title('ARL vs. Batch Size')
    plt.savefig("./Figures/arl_0_vs_batch_violin.png", dpi=500)
    plt.show()
    plt.close()


def plot_facet_batch_grid(df):
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="Batch Size", palette="tab20c",
                         col_wrap=3)

    # pass additional arguments
    grid = grid.map(plt.plot, "Threshold", "ARL_0",  marker='.')
    plt.savefig("./Figures/Facet_Grid_ARL_0-vs-h_t_v2.png", dpi=800)
    # Adjust the arrangement of the plots
    # grid.fig.tight_layout(w_pad=1)


def plot_facet_threshold_grid(df):
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="Threshold", palette="tab20c",
                         col_wrap=3)

    # pass additional arguments
    grid = grid.map(plt.plot, "Batch Size", "ARL_0", marker='.')
    plt.savefig("./Figures/Facet_Grid_ARL_0-vs-batch_v2.png", dpi=800)
    # Adjust the arrangement of the plots
    # grid.fig.tight_layout(w_pad=1)


def semilogx_scatter(x, y, **kwargs):
    ax = plt.gca()
    sns.scatterplot(x, y, **kwargs)
    ax.set_xscale('log')


def plot_hist_prob(y, **kwargs):
    ax = plt.gca()
    plt.hist(y, **kwargs)
    plt.xlim(0, 1)


def plot_facet_semilogx_grid(df, file_name):
    """

    """
    g = sns.FacetGrid(df, row="ARL_0", col="Batch Size", hue="rho", margin_titles=True)
    g.map(semilogx_scatter, "delta_rho", "ARL_1")
    g.add_legend()
    plt.savefig(file_name, dpi=900)
    plt.show()
    plt.close()


def plot_facet_kde(df, file_name, row_name="ARL_0", target="Conditional_Correct_Detection"):
    """
    Plot a facet grid of the distribution of correct detection given no false alarm
    """
    print("Saving " + file_name)
    g = sns.FacetGrid(df, row=row_name, margin_titles=True)
    g.map(sns.distplot, target, kde=True, rug=True, norm_hist=True)
#    g.map(plt.hist, target, density=True, stacked=True)
    plt.savefig(file_name, dpi=900)
    plt.show()
    plt.close()


def plot_facet_hist_prob(df, file_name, row_name="ARL_0", col_name=None, target="Conditional_Correct_Detection"):
    """
    Plot a Facet grid of a correct or conditionally correct detection probability
    """
    print("Saving " + file_name)
    if col_name:
        g = sns.FacetGrid(df, row=row_name, col=col_name, margin_titles=True)
    else:
        g = sns.FacetGrid(df, row=row_name, margin_titles=True)
    g.map(plot_hist_prob, target, density=True, stacked=True)
    plt.savefig(file_name, dpi=900)
    plt.show()
    plt.close()



def plot_facet_semilogx_conditional_correct_detection(df, file_name):
    """
    Plot the probability of correct detection given no false alarm
    """
    g = sns.FacetGrid(df, row="ARL_0", col="Batch Size", hue="rho", margin_titles=True)
    g.map(semilogx_scatter, "delta_rho", "Conditional_Correct_Detection")
    g.add_legend()
    plt.savefig(file_name, dpi=900)
    plt.show()
    plt.close()


def plot_facet_semilogx_correct_detection(df, file_name):
    """
    Plot the probability of correct detection given no false alarm
    """
    g = sns.FacetGrid(df, row="ARL_0", col="Batch Size", hue="rho", margin_titles=True)
    g.map(semilogx_scatter, "delta_rho", "Correct_Detection")
    g.add_legend()
    plt.savefig(file_name, dpi=900)
    plt.show()
    plt.close()


def plot_three_d_arl0_rho_delta_rho(df, file_prefix, batch):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['rho'], df['delta_rho'], df['ARL_1'], c=df["ARL_0"], s=60)
    file_name = file_prefix + "batch_{}.png".format(batch)
    ax.view_init(60, 35)
    ax.set_xlabel("rho")
    ax.set_ylabel("delta rho")
    ax.set_zlabel("ARL_1")
    plt.title("Batch {}".format(batch))
    plt.savefig(file_name, dpi=900)
    plt.show()


def plot_three_d_plots(detection_delay_df, file_prefix):
    # plot for each batch size:
    batches = list(detection_delay_df["Batch Size"].unique())
    for batch in batches:
        plot_three_d_arl0_rho_delta_rho(detection_delay_df[detection_delay_df["Batch Size"] == batch],
                                        file_prefix, batch)


def plot_facet_grid(df, file_name):
    """

    """
    g = sns.FacetGrid(df, row="ARL_0", col="Batch Size", hue="rho", margin_titles=True)
    g.map(plt.scatter, "delta_rho", "ARL_1")
    g.add_legend()
    plt.savefig(file_name, dpi=900)
    plt.show()
    plt.close()


def plot_three_d_arl0_rho_delta_rho(df, file_prefix, batch):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['rho'], df['delta_rho'], df['ARL_1'], c=df["ARL_0"], s=60)
    file_name = file_prefix + "batch_{}.png".format(batch)
    ax.view_init(30, 185)
    plt.savefig(file_name, dpi=900)
    plt.show()


def plot_three_d_plots(detection_delay_df, file_prefix):
    # plot for each batch size:
    batches = list(detection_delay_df["Batch Size"].unique())
    for batch in batches:
        plot_three_d_arl0_rho_delta_rho(detection_delay_df, file_prefix, batch)


def plot_two_d_arl0_rho_delta_rho(detection_delay_df, file_prefix, batch):
    file_name = file_prefix + "batch_{}.png".format(batch)
    sns.scatterplot(x="delta_rho", y="ARL_1", hue="ARL_0", style="delta_rho", data=detection_delay_df)
    plt.savefig(file_name, dpi=900)
    plt.close()


def plot_two_d_plots(detection_delay_df, file_prefix):
    # plot for each batch size:
    batches = list(detection_delay_df["Batch Size"].unique())
    for batch in batches:
        plot_two_d_arl0_rho_delta_rho(detection_delay_df, file_prefix, batch)


def main():
    batch_threshold_arl_dict = load_pkl_data()
    batch_threshold_arl_df = pd.DataFrame(data=batch_threshold_arl_dict)
    # Give me a quick synopsis of the data
    batch_threshold_arl_df.describe()

    # Plot a violin plot
    plot_violin_data_type_1(batch_threshold_arl_df)
    plot_violin_data_type_2(batch_threshold_arl_df)
    plot_violin_data_type_3(batch_threshold_arl_df)

    # plot a facet grid
    plot_facet_batch_grid(batch_threshold_arl_df)
    plot_facet_threshold_grid(batch_threshold_arl_df)


def load_power_df(file_name):
    data = PowerTestLogger(file_name, is_full_path=True, is_dataframe=True)
    data_df = data.load_dataframe()
    return data_df


def load_all_data():
    log_directory = "./Results/GLRT_ROSS/ARL_1/"
    log_file_name = log_directory + "detection_delay_test_log_"
    log_file_name_1 = log_file_name + "06_05_20.bz2"
    log_file_name_2 = log_file_name + "06_07_20.bz2"
    log_file_name_3 = log_file_name + "06_25_20.bz2"
    log_file_name_4 = log_file_name + "06_01_20.bz2"
    log_file_name_5 = log_file_name + "05_31_20.bz2"
    log_file_name_6 = log_file_name + "06_18_20.bz2"
    log_file_name_7 = log_file_name + "06_24_20.bz2"
    log_file_name_8 = log_file_name + "05_29_20..bz2"
    log_file_name_9 = log_file_name + "05_30_20..bz2"
    # load the data
    data_df_1 = load_power_df(log_file_name_1)
    data_df_2 = load_power_df(log_file_name_2)
    data_df_3 = load_power_df(log_file_name_3)
    data_df_4 = load_power_df(log_file_name_4)
    data_df_5 = load_power_df(log_file_name_5)
    data_df_6 = load_power_df(log_file_name_6)
    data_df_7 = load_power_df(log_file_name_7)
    data_df_8 = load_power_df(log_file_name_8)
    data_df_9 = load_power_df(log_file_name_9)
    frames = [data_df_1, data_df_2, data_df_3, data_df_4, data_df_5, data_df_6, data_df_7, data_df_8, data_df_9]
    total_df = pd.concat(frames, axis=0, ignore_index=True, verify_integrity=True)
    #     total_df["Batch Size"] = pd.to_numeric(total_df["Batch Size"])
    #total_df = total_df[total_df["Batch Size"].isin(selected_batches)]
#    total_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#    total_df.dropna(inplace=True)
    return total_df


def main_jp_test():
    """
    Results for uniformly distributed run lengths.
    Plotting the probability of a first detection that is correct
    and
    probability of a correct detection
    """
    total_df = load_all_data()
    # plot_facet_semilogx_conditional_correct_detection(total_df, "./Figures/ARL_1/JP_Facet_cond_correct.png")
    # plot_facet_semilogx_correct_detection(total_df, "./Figures/ARL_1/JP_Facet_correct.png")

    # Plot combinatins of rho, delta_rho, ARL_0 and batch size
    figure_prefix = "./Figures/ARL_1/FacetHistogram"
#     plot_facet_kde(total_df, figure_prefix + "_rho_delta_rho_3.png", row_name="rho", target="Correct_Detection")
#     plot_facet_kde(total_df, figure_prefix + "_rho_batch_size_3.png", row_name="Batch Size", target="Correct_Detection")
#     plot_facet_kde(total_df, figure_prefix + "_arl_0_batch_size_3.png", row_name="ARL_0", target="Correct_Detection")
#     figure_prefix = "./Figures/ARL_1/FacetHistogram_CondCorrectDetection"
#     plot_facet_kde(total_df, figure_prefix + "_rho_delta_rho_3.png", row_name="rho",
#                    target="Conditional_Correct_Detection")
#     plot_facet_kde(total_df, figure_prefix + "_rho_batch_size_3.png", row_name="Batch Size",
#                    target="Conditional_Correct_Detection")
#     plot_facet_kde(total_df, figure_prefix + "_arl_0_batch_size_3.png", row_name="ARL_0",
#                    target="Conditional_Correct_Detection")
# #     plot_facet_grid(total_df, figure_prefix + "_full_grid_3.png")
    plot_facet_hist_prob(total_df, figure_prefix + "_ARL_0_conditional_correct_detection_3.png",
                         row_name="ARL_0", col_name="Batch Size")
    plot_facet_hist_prob(total_df, figure_prefix + "_ARL_0_correct_detection_3.png",
                         row_name="ARL_0", col_name="Batch Size", target="Correct_Detection")
    plot_facet_hist_prob(total_df, figure_prefix + "_Batch_Size_ARL_0_ARL_1_3.png",
                         row_name="ARL_0", col_name="Batch Size", target="ARL_1")


#     total_df.to_csv("./Results/GLRT_ROSS/ARL_1/arl_1_data_full.csv")


def main_goldsman_test():
    """
    Results for uniformly distributed run lengths.
    Plotting the probability of a correct detection for different run lengths
    """
#     log_directory = "./Results/GLRT_ROSS/ARL_1/"
#     log_file_name = log_directory + "Goldsman_test_log_06_11_20.bz2"
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/"
    log_file_name = log_directory + "run_length_test_log_07_05_20.bz2"
    data = PowerTestLogger(log_file_name, is_full_path=True, is_dataframe=True)
    data_df = data.load_dataframe()
    print("Columns: ", data_df.columns)
    figure_prefix = "./Figures/ARL_1/Facet_Run_Length"
    plot_facet_hist_prob(data_df, figure_prefix + "_run_length_correct_detection_rho_3.png",
                         col_name="rho",
                         row_name="Run Length", target="Correct_Detection")
    plot_facet_hist_prob(data_df, figure_prefix + "_run_length_cond_correct_detection_rho_3.png", row_name="Run Length",
                         col_name="rho",
                         target="Conditional_Correct_Detection")


def main_plot_fixed_slice():
    """
        Free one variable at a time and plot the
    """
    total_df = load_all_data()
    delta_rho_list = list(total_df["delta_rho"].unique())
    ARL_0 = 150000
    delta_rho = delta_rho_list[0]
#     total_df = total_df[total_df["ARL_0"] == ARL_0]
    total_df = total_df[total_df["delta_rho"] == delta_rho]
    rho_list = list(total_df["rho"].unique())
    fig_prefix = "./Figures/ARL_1/Fixed_JP/"
    idx = 0
    for rho in rho_list:
        idx += 1
        fig_name = fig_prefix + "correct_detection_vs_batch_delta_rho_fixed_rho_{}.png".format(idx)
        limited_df = total_df[total_df["rho"] == rho]
        sns.scatterplot(data=limited_df, x="Batch Size", y="Correct_Detection", hue="delta_rho")
        plt.xlabel("Batch Size")
        plt.ylabel("Correct Detection")
        plt.title("rho={0:2.4f}".format(rho))
        plt.savefig(fig_name, dpi=800)
        plt.show()

def plot_simple_scatterplot(fig_prefix, selected_df, rho, ARL_0_val, y_col="Correct_Detection"):
    sns.scatterplot(x='Batch Size', y=y_col, hue='delta_rho', data=selected_df)
    plt.title("rho={}, ARL_0={}".format(rho, ARL_0_val))
    L = plt.legend()
    print(L)
    print(L.get_texts())
    print(L.get_texts()[0], L.get_texts()[1], L.get_texts()[2])
    L.get_texts()[1].set_text('25%')
    L.get_texts()[2].set_text('50%')
    L.get_texts()[3].set_text('75%')
    L.get_texts()[4].set_text('100%')
    plt.savefig(fig_prefix + "for_rho{}.png".format(rho), dpi=900)
    plt.show()
    plt.close()


def facet_scatterplot(fig_prefix, selected_df, ARL_0, y_col = "Correct_Detection"):
    g = sns.FacetGrid(selected_df, row='Time of Change', col="rho", hue="delta_rho", margin_titles=True)
    g.map(sns.scatterplot, "Batch Size", y_col)
    g.add_legend()
    plt.title("ARL_0={}".format( ARL_0))
    plt.savefig(fig_prefix + ".png", dpi=900)
    plt.close()


def load_and_plot_new_jp_test():
    """
    Created this function on July 6th to represent the last results of the sim
    """
    log_directory = "./Results/GLRT_ROSS/ARL_1/"
    # log_file_name = log_directory + "detection_delay_test_log_07_06_20.bz2"
    log_file_name = log_directory + "select_detection_delay_test_log_07_10_20.bz2"
    fig_prefix_1 = "./Figures/ARL_1/Fixed_JP/Results_week2/"
    fig_prefix_2 = "CorrectDetection_BatchSize_Different_ARL0_Different_delta_rho_"
    rho_list = list()
    data_df = load_power_df(log_file_name)
    rho_list = list(data_df["rho"].unique())
    ARL_0_val = list(data_df["ARL_0"].unique())[0]
#     for rho in rho_list:
    # selected_df = data_df[data_df["rho"] == rho]
    selected_df = data_df
    # Conditional Correct Detection
    fig_prefix = fig_prefix_1 + "Cond" + fig_prefix_2
    facet_scatterplot(fig_prefix, selected_df, ARL_0_val, "Conditional_Correct_Detection")
    fig_prefix = fig_prefix_1 + fig_prefix_2
    facet_scatterplot(fig_prefix, selected_df, ARL_0_val)


def load_and_plot_jp_test_display_less_data():
    """
    Created this function on July 15th to present other results in a clearer way.
    """
    fig_prefix_1 = "./Figures/ARL_1/Fixed_JP/Results_week2/"
    log_directory = "./Results/GLRT_ROSS/ARL_1/"
    log_file_name = log_directory + "select_detection_delay_test_log_07_10_20.bz2"
    data_df1 = load_power_df(log_file_name)
    # data_df1 = data_df1[data_df1["Conditional_Correct_Detection"] > 0.0]
    # Open second log file
    log_file_name_2 = log_directory + "select_detection_delay_test_log_07_15_20.bz2"
    data_df2 = load_power_df(log_file_name_2)
    frames = [data_df1, data_df2]
    data_df = pd.concat(frames, axis=0, ignore_index=True, verify_integrity=True)
    delta_rho_list = list(data_df["delta_rho"].unique())
    time_of_change_list = list(data_df['Time of Change'].unique())
    rho_list = list(data_df["rho"].unique())
    # Pick rho=0.75
    for rho_selected in rho_list:
        modified_df = data_df[data_df["rho"] == rho_selected]
        for time_of_change in time_of_change_list:
            fig_prefix_2 = "CondCorrectDetection_BatchSize_rho_point{}_Tchange_{}".format(
                rho_selected * 100, time_of_change
            )
            specific_df = modified_df[modified_df["Time of Change"] == time_of_change]
            fig_name = fig_prefix_1 + fig_prefix_2 + ".png"
            if len(specific_df) < 2:
                continue
            print("Rho={0} and Tchange={1}".format(rho_selected, time_of_change))
            g = sns.FacetGrid(data=specific_df, col="delta_rho", col_wrap=2, sharey=True, ylim={0, 1})
            g = g.map(sns.scatterplot, "Batch Size", "Conditional_Correct_Detection")
            g.add_legend()
            if rho_selected < 0.75:
                plt.subplots_adjust(top=0.9)
            else:
                plt.subplots_adjust(top=0.8)
            g.fig.suptitle("rho={}, T_change={}".format(rho_selected, time_of_change))
            plt.savefig(fig_name, dpi=888)
            # plt.show()
            plt.close()

#         for delta_rho in delta_rho_list:
#             specialized_df = data_df[data_df["delta_rho"] == delta_rho]
#             for time_of_change in time_of_change_list:
#                 specific_df = specialized_df[specialized_df["Time of Change"] == time_of_change]
#                 sns.scatterplot(x='Batch Size', y="Conditional_Correct_Detection", data=specific_df)
#                 plt.title("rho={}, delta_rho/rho={}, T_change={}".format(rho_selected, delta_rho, time_of_change))
#                 plt.ylim([0, 1.0])
#                 plt.savefig(fig_name, dpi=888)
#                 plt.close()

# Iterate through delta_rho/rho


def plot_signal_noise_ratio_effect():
    """
    Plot the effects of the noise on the signal
    """
    log_directory = "./Results/GLRT_ROSS/ARL_1/"
    log_file_name = log_directory + "select_detection_delay_test_log_07_10_20.bz2"
    data_df1 = load_power_df(log_file_name)

    # Open second log file
    log_file_name_2 = log_directory + "select_detection_delay_test_log_07_15_20.bz2"
    data_df2 = load_power_df(log_file_name_2)
    frames = [data_df1, data_df2]
    data_df = pd.concat(frames, axis=0, ignore_index=True, verify_integrity=True)
    data_df.loc[:, "Noise"] = np.sqrt(data_df["rho"])

    # Plot 1 Signal vs Noise
    sns.relplot(x="Noise", y="delta_rho", size="Batch Size", data=data_df)
    plt.show()
    # Plot 2 Best Batch size vs. Signal for different change times

    # Plot 3 conditional probability vs signal for different change times


def examine_generalization_results():
    """
    Test the results from fitting the probability of correct detection

    """
    # 0. read pickle file
    pickle_file_name = "best_fit.pkl"
    data_df = pd.read_pickle(pickle_file_name)
    data_df = data_df[data_df["Fit Type"] == "fisk"]
    data_df["SNR"] = data_df["delta_rho"] / np.sqrt(data_df["rho"])
    data_df["rho_ratio"] = data_df["delta_rho"] / data_df["rho"]
    # Create a facetgrid of the best batch size vs SNR for different time of changes
    grid = sns.FacetGrid(data_df, hue="Time_of_Change", col="rho",
                         col_wrap=4)
    # pass additional arguments
    grid = grid.map(plt.scatter, "SNR", "Batch Size",  marker='.')
    grid.add_legend()
#    plt.title("Log-normal")
    plt.savefig("log_logistic_batch_vs_snr_rho_variation_col_rho_hue_tchange.png", dpi=1000)
    plt.show()


if __name__ == "__main__":
    #main_jp_test()
#     main_goldsman_test()
#     main_plot_fixed_slice()
    # load_and_plot_new_jp_test()
#    load_and_plot_jp_test_display_less_data()
    #plot_signal_noise_ratio_effect()
    examine_generalization_results()
