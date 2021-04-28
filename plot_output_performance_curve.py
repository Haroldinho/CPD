import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def main_0(test_results_df):
    for selected_time_of_change in list(test_results_df["Time of change"].unique()):
        reduced_df = test_results_df[test_results_df["Time of change"] == selected_time_of_change]
        sns.relplot(data=reduced_df, x="Recall", y="Precision", hue="Time of change", size="Batch Size",
                    alpha=.5, palette="muted", height=6)
        plt.savefig(f"Precision_vs_Recall_{selected_time_of_change}.png")
        plt.close()
        sns.relplot(data=reduced_df, x="fa_rate", y="tp_rate", hue="ARL_0", size="Set Autocorrelation",
                    alpha=.5, palette="muted", height=6)
        diag_x = np.linspace(0, 1, 100)
        diag_y = diag_x
        plt.plot(diag_x, diag_y, '-.')
        plt.savefig(f"TP_vs_FA_{selected_time_of_change}.png")
        plt.close()
        sns.relplot(data=reduced_df, x="fa_rate", y="Correct_Detection", hue="ARL_0", size="Set Autocorrelation",
                    alpha=.5, palette="muted", height=6)
        diag_x = np.linspace(0, 1, 100)
        diag_y = diag_x
        plt.plot(diag_x, diag_y, '-.')
        plt.savefig(f"CorrectDetection_vs_FA_{selected_time_of_change}.png")
        plt.close()


def main_1(test_results_df):
    print("working on test_results_df")
    sns.relplot(data=test_results_df, x="Recall", y="Precision", size="Batch Size",
                alpha=.5, palette="muted", height=6)
    plt.savefig("Precision_vs_Recall.png")
    plt.close()
    sns.relplot(data=test_results_df, x="fa_rate", y="tp_rate", hue="ARL_0", size="Set Autocorrelation",
                alpha=.5, palette="muted", height=6)
    diag_x = np.linspace(0, 1, 100)
    diag_y = diag_x
    plt.plot(diag_x, diag_y, '-.')
    plt.savefig("TP_vs_FA.png")
    plt.close()
    sns.relplot(data=test_results_df, x="fa_rate", y="Correct_Detection", hue="ARL_0", size="Set Autocorrelation",
                alpha=.5, palette="muted", height=6)
    diag_x = np.linspace(0, 1, 100)
    diag_y = diag_x
    plt.plot(diag_x, diag_y, '-.')
    plt.savefig("CorrectDetection_vs_FA.png")
    plt.close()


def main_2(test_results_df):
    print("working on test_results_df")
    sns.relplot(data=test_results_df, x="Recall", y="Precision", hue="delta_rho",
                alpha=.5, palette="muted", height=6)
    plt.title("rho=0.5")
    plt.savefig("Precision_vs_Recall_delta_rho.png")
    plt.close()
    sns.relplot(data=test_results_df, x="fa_rate", y="tp_rate", hue="delta_rho", size="Set Autocorrelation",
                alpha=.5, palette="muted", height=6)
    diag_x = np.linspace(0, 1, 100)
    diag_y = diag_x
    plt.title("rho=0.5")
    plt.plot(diag_x, diag_y, '-.')
    plt.savefig("TP_vs_FA_delta_rho.png")
    plt.close()
    sns.relplot(data=test_results_df, x="fa_rate", y="Correct_Detection", hue="delta_rho", size="Set Autocorrelation",
                alpha=.5, palette="muted", height=6)
    diag_x = np.linspace(0, 1, 100)
    diag_y = diag_x
    plt.title("rho=0.5")
    plt.plot(diag_x, diag_y, '-.')
    plt.savefig("CorrectDetection_vs_FA_delta_rho.png")
    plt.close()


class DeltaRhoToMarkerSizeTransformer:
    def __init__(self, delta_rho_list):
        self.list_of_delta_rhos = sorted(delta_rho_list)
        self._size_list_map = {delta_rho: 20 * 2 ** i for i, delta_rho in enumerate(self.list_of_delta_rhos)}

    def convert_delta_rho_to_marker_size(self, delta_rho):
        return self._size_list_map[delta_rho]


def plot_hypothesis_vals_vs_delta_rhos(results_df, batch_size, is_nonparametric=False):
    results_df.loc[results_df["delta_rho"] == 0.0, "TP"] = np.nan
    results_df.loc[results_df["delta_rho"] == 0.0, "FP"] = np.nan
    results_df.loc[results_df["delta_rho"] == 0.0, "TN"] = np.nan
    results_df.loc[results_df["delta_rho"] == 0.0, "FN"] = np.nan
    plt.scatter(results_df["delta_rho"], results_df["TP"], c="k", marker="+", label="True Positive")
    plt.scatter(results_df["delta_rho"], results_df["FP"], c="b", marker="d", label="False Positive")
    plt.scatter(results_df["delta_rho"], results_df["FN"], c="g", marker="*", label="False Negative")
    plt.scatter(results_df["delta_rho"], results_df["TN"], c="c", marker=".", label="True Negative")
    plt.xlabel("Signal Strength [Delta_Rho]")
    plt.ylabel("Results")
    axes = plt.gca()
    axes.set_ylim([0, 1000])
    plt.legend(loc="best")
    if is_nonparametric:
        plt.savefig(f"Delta_Rho_HypothesisTestResults_batch_{batch_size[0]}_nonparametric.png", dpi=800)
    else:
        plt.savefig(f"Delta_Rho_HypothesisTestResults_batch_{batch_size[0]}.png", dpi=800)
    plt.close()


def plot_precision_vs_recall(master_df, is_nonparametric=False):
    plt.figure()
    if "Batch Size" in master_df.columns:
        batch_size = master_df["Batch Size"].unique()
    else:
        batch_size = None
    sns.relplot(data=master_df, x="Recall", y="Precision", hue="delta_rho", alpha=.5, height=6)
    plt.title("rho=0.5")
    if is_nonparametric:
        plt.savefig(f"Precision_vs_Recall_delta_rho_batch_{batch_size[0]}_nonparametric.png")
    else:
        plt.savefig(f"Precision_vs_Recall_delta_rho_batch{batch_size[0]}.png")
    plt.close()


def scatter_text(x, y, text_column, data, title, xlabel, ylabel):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, size=8, legend=False)
    # Add text besides each point
    axes = plt.gca()
    #     axes.set_xlim([0, 1])
    #     axes.set_ylim([0, 1])
    x_window_size = data[x].max() - data[x].min()
    offset = 0.0001 * x_window_size
    for line in range(0, data.shape[0]):
        posx = data[x][line] + offset
        posy = data[y][line]
        text = data[text_column][line]
        p1.text(posx, posy, text, horizontalalignment='left', fontsize=10, color='black', weight='semibold')
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1


def plot_roc_curve(master_df, is_nonparametric=False):
    #     plt.figure()
    if "Batch Size" in master_df.columns:
        batch_size = master_df["Batch Size"].unique()
    else:
        batch_size = None
    #     sns.relplot(data=master_df, x="fp_rate", y="tp_rate", hue="delta_rho", alpha=.5, height=6)
    #
    #     x_points = list(master_df["fp_rate"])
    #     y_points = list(master_df["tp_rate"])
    #     label_points = list(master_df["delta_rho"])
    #
    #     plt.title("rho=0.5")
    master_df.dropna(inplace=True)
    scatter_text("fp_rate", "tp_rate", "delta_rho", data=master_df, title=f"ROC {batch_size[0]}", xlabel="FP rate",
                 ylabel="TP rate")
    if is_nonparametric:
        plt.savefig(f"ROC_rho_batch_{batch_size[0]}_nonparametric.png")
    else:
        plt.savefig(f"ROC_rho_batch_{batch_size[0]}_parametric.png")
    plt.close()


def main_simple_test(results_df, is_nonparametric=False):
    batch_size = results_df["Batch Size"].unique()
    delta_rhos = results_df["delta_rho"].unique()
    # plot_precision_vs_recall(results_df, is_nonparametric)
    parametric_text = "non-parametric" if is_nonparametric else "parametric"
    print(f"working on  batch {batch_size} which is {parametric_text}")
    #     sns.relplot(data=results_df, x="Recall", y="Precision", hue="delta_rho", size="h_t",
    #                 alpha=.5, palette="muted", height=6)
    plot_hypothesis_vals_vs_delta_rhos(results_df.copy(), batch_size, is_nonparametric)
    plot_roc_curve(results_df.copy(), is_nonparametric)
    # make the size depends on delta_rho
    # for each delta_rho gets the next size


#     transformer = DeltaRhoToMarkerSizeTransformer(delta_rhos)
#     vec_of_delta_rhos = [delta_rho for delta_rho in results_df["delta_rho"]]
#     vec_of_marker_size = list(map(transformer.convert_delta_rho_to_marker_size, vec_of_delta_rhos))
#     print(vec_of_marker_size)
#     plt.scatter(results_df["h_t"], results_df["TP"], c="k", marker='+', s=vec_of_marker_size, label="True Positive")
#     plt.scatter(results_df["h_t"], results_df["FP"], c="b", marker='.', s=vec_of_marker_size, label="False Positive")
#     plt.scatter(results_df["h_t"], results_df["FN"], c="g", marker='d', s=vec_of_marker_size, label="False Negative")
#     plt.scatter(results_df["h_t"], results_df["TN"], c="c", marker='*', s=vec_of_marker_size, label="True Negative")
#     plt.xlabel("Threshold")
#     plt.ylabel("Results")
#     plt.legend(loc="best")
#     plt.savefig(f"HypothesisTestResults_batch_{batch_size[0]}.png", dpi=800)
#     plt.close()
#     for delta_rho in delta_rhos:
#         reduced_df = results_df[results_df["delta_rho"] == delta_rho]
# #         sns.relplot(data=reduced_df, x="fp_rate", y="tp_rate", hue="h_t",
# #                     alpha=.5, palette="muted", height=6)
#         sns.relplot(data=reduced_df, x="fp_rate", y="tp_rate",
#                     alpha=.5, palette="muted", height=6)
#         diag_x = np.linspace(0, 1, 100)
#         diag_y = diag_x
#         plt.title(f"ROC\nrho=0.5, delta rho={delta_rho}, Batch Size={batch_size[0]}")
#         plt.plot(diag_x, diag_y, '-.')
#         if is_nonparametric:
#             plt.savefig(f"TP_vs_FA_delta_rho_{delta_rho}_batch_{batch_size[0]}_nonparametric.png")
#         else:
#             plt.savefig(f"TP_vs_FA_delta_rho_{delta_rho}_batch_{batch_size[0]}.png")
#         plt.close()
# #         sns.relplot(data=reduced_df, x="fp_rate", y="Correct_Detection", hue="h_t",
# #                     alpha=.5, palette="muted", height=6)
#         sns.relplot(data=reduced_df, x="fp_rate", y="Correct_Detection",
#                     alpha=.5, palette="muted", height=6)
#         diag_x = np.linspace(0, 1, 100)
#         diag_y = diag_x
#         txt = "non-parametric" if is_nonparametric else "parametric"
#         plt.title(f"rho=0.5, delta rho={delta_rho}, Batch Size={batch_size[0]} - {txt}")
#         if is_nonparametric:
#             plt.savefig(f"CorrectDetection_vs_FA_delta_rho_{delta_rho}_batch_{batch_size[0]}_nonparametric.png")
#         else:
#             plt.savefig(f"CorrectDetection_vs_FA_delta_rho_{delta_rho}_batch_{batch_size[0]}.png")
#         plt.close()


def plot_histograms_of_correct_and_incorrect_detection(test_results_df):
    """
    Plot different histogram of correct and incorrect detections for all the scores
    Set a point as a correct detection if TP + FP > TN + FN
    """
    test_results_df["NumberOfDetections"] = test_results_df["TP"] + test_results_df["FP"]
    test_results_df["NumberOfNoDetections"] = test_results_df["TN"] + test_results_df["FN"]
    test_results_df["MoreDetections"] = test_results_df["NumberOfDetections"] - test_results_df["NumberOfNoDetections"]
    test_results_df["IsCorrect"] = test_results_df.MoreDetections > 0
    sns.displot(test_results_df, x="G_n", hue="IsCorrect")
    plt.savefig("DistributionofDetections.png")
    plt.close()


def plot_density_histogram_of_detections(is_true_vec, score_vec, title=None):
    data = pd.DataFrame.from_dict({"TrueChange": is_true_vec, "Score": score_vec})
    sns.displot(data, x="Score", hue="TrueChange")
    if title:
        plt.title(title)
    plt.savefig("DistributionOfDetections.png", dpi=800)
    plt.close()


if __name__ == "__main__":
    os.chdir("Results/GLRT_ROSS/Performance_Tests")
    print(os.listdir())
    # test_results_df = pd.read_csv("SingleDetection.csv")
    #    results_df = pd.read_csv("SimpleDetection_BATCH_OF_SIZE_5.csv")
    #     results_df = pd.read_csv("SimpleDetection_Batch_of_size_50_rcpm.csv")
    #     main_simple_test(results_df)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_50_rcpm_nonparam.csv")
    main_simple_test(results_df, True)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_100_rcpm_nonparam.csv")
    main_simple_test(results_df, True)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_150_rcpm_nonparam.csv")
    main_simple_test(results_df, True)

    results_df = pd.read_csv("SimpleDetection_Batch_of_size_50_rcpm.csv")
    main_simple_test(results_df)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_100_rcpm.csv")
    main_simple_test(results_df)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_150_rcpm.csv")
    main_simple_test(results_df)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_200_rcpm.csv")
    main_simple_test(results_df)
    results_df = pd.read_csv("SimpleDetection_Batch_of_size_500_rcpm.csv")
    main_simple_test(results_df)
