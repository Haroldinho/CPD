import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
from batch_selection import BatchSelectorOnAutocorrelation
from average_run_length_loader import ThresholdSelector


def get_correlation_batch_size_heat_map(target_rhos, target_correlations):
    external_rt_dict = dict()
    for corr in target_correlations:
        batch_selector = BatchSelectorOnAutocorrelation(corr)
        batch_list = []
        for rho in target_rhos:
            batch_list.append(batch_selector.return_batch_size(rho))
        external_rt_dict[corr] = batch_list
    correlation_batch_size_df = pd.DataFrame(external_rt_dict, columns=target_correlations, index=rhos)
    ax = sns.heatmap(correlation_batch_size_df, annot=True, norm=LogNorm(), cbar=False)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    plt.title("Heatmap of Batch Size ")
    plt.xlabel("Lag-1 auto-correlation")
    plt.ylabel("rho")
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    plt.savefig("AutoCorrelation_BatchSize_Heatmap.png", dpi=900)
    plt.show()


def get_threshold_arl_0_heat_map(target_arl_0, my_batch_sizes):
    pkl_directory = "./Results/GLRT_ROSS/ARL_0/"
    h_selector = ThresholdSelector(pkl_directory)
    external_arl_0_dic = {}
    for arl_0 in target_arl_0:
        h_t_list = []
        for batch in my_batch_sizes:
            h_t_list.append(h_selector.get_threshold(batch, arl_0))
        external_arl_0_dic[arl_0] = h_t_list
    print(external_arl_0_dic)
    arl_0_detection_threshold_df = pd.DataFrame(external_arl_0_dic, columns=target_arl_0, index=my_batch_sizes)
#    arl_0_detection_threshold_df[arl_0_detection_threshold_df < 0] = 0
    arl_0_detection_threshold_df = arl_0_detection_threshold_df.astype(float)
    arl_0_detection_threshold_df.to_csv("arl_0_detection_threhsold_matrix.csv")
    ax = sns.heatmap(arl_0_detection_threshold_df, annot=True)
    plt.title("Heatmap of Detection Threshold vs. (Batch Size x ARL0)")
    plt.savefig("ARL_0_Threshold_Heatmap.png", dpi=900)
    plt.show()


if __name__ == "__main__":
    desired_correlations = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    rhos = np.linspace(0.1, 0.9, 9)
    batch_sizes = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
    arl_0_vector = [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 1e4, 1e5]
    get_correlation_batch_size_heat_map(rhos, desired_correlations)
#    get_threshold_arl_0_heat_map(arl_0_vector, batch_sizes)
