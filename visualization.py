"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: visualization.py
Description: methods to visualize the different outputs of the queueing simulations and detection procedures


Author: Harold Nemo Adodo Nikoue
part of the partial observability thesis
"""

from typing import List

import numpy as np
from matplotlib import pyplot as plt


def plot_wait_times(wait_times_vector: List[float], arrival_rate: float, service_rate: float, file_name: str):
    plt.figure()
    plt.plot(wait_times_vector)
    plt.xlabel("n")
    plt.ylabel("Wait times vs. n")
    plt.title("Arr. Rate={}, Serv. Rate={} ".format(arrival_rate, service_rate))
    plt.savefig(file_name, dpi=500)
    plt.show()


def plot_ladder_points(ladder_times: List[float], arrival_rate: float, service_rate: float, file_name: str):
    plt.plot(ladder_times)
    plt.xlabel("n")
    plt.ylabel("S_n")
    plt.title("Arr. Rate={0}, Serv. Rate={1} ".format(arrival_rate, service_rate))
    plt.savefig(file_name, dpi=500)
    plt.show()


def plot_score_vs_n(time_abscissa, score_ordinate, x_name, y_name, title, file_name):
    plt.plot(time_abscissa, score_ordinate, 'ro')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()


def plot_wait_times_against_D_k_n(wait_times: List[float], associated_times: List[float],
                                  detected_times: List[float], detection_stats: List[float], threshold: float,
                                  change_point, file_name: str):
    """
    Use this function to plot the wait times on one axis,
    the detection statistics and detection threshold on the second axis
    Both wait times and detection statistics are plotted against the same time axis.
    :param wait_times:
    :param associated_times:
    :param detected_times:
    :param detection_stats:
    :param threshold:
    :param change_point:
    :param file_name:
    :return: nothing
    """
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    plt.vlines(change_point, min(wait_times), max(wait_times), linestyles="dashed")
    plt.plot(associated_times, wait_times, '-.', color=color)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wait Times", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('D_k_n', color=color)
    ax2.plot(detected_times, detection_stats, 'o', color=color)
    plt.hlines(threshold, min(associated_times), max(associated_times), colors='red')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(file_name)
    plt.show()


def plot_wait_times_with_detected_changes(wait_times: List[float], actual_times: List[float],
                                          detected_indices: List[int], threshold: float, change_point,
                                          file_name: str):
    """
    This function was used with the Kingman's bound which provided a bound on the Ladder heights and therefore weight
    times directly.
    DO NOT USE THIS FUNCTION TO PLOT THE DETECTION THRESHOLD FOR A LIKELIHOOD BASED TEST.
    Instead use
    :wait_times:
    """
    detected_wait_times = [wait_times[idx] for idx in detected_indices]
    detected_times = [actual_times[idx] for idx in detected_indices]
    plt.plot(actual_times, wait_times, 'k')
    plt.plot(detected_times, detected_wait_times, 'r*')
    plt.plot(actual_times, [threshold] * len(actual_times), 'm')
    plt.vlines(change_point, min(wait_times), max(wait_times), linestyles="dashed")
    plt.xlabel('Time')
    plt.ylabel('Wait Times')
    plt.savefig(file_name)
    plt.show()


def plot_power_test_heatmap(detection_delay_power_mat, significance_levels, thresholds, batch_size, file_name):
    fig, ax = plt.subplots()
    heat_map = ax.imshow(detection_delay_power_mat)
    plt.xlabel('significance level')
    plt.ylabel('Thresholds')
    plt.title("Heat map for a batch size of size {}".format(batch_size))
    plt.savefig(file_name, dpi=800)
    plt.show()


def plot_power_test_2_heatmap(full_stat_structure, batch_sizes, rhos, delta_rhos, file_prefix, detection_threshold,
                              cbar_name, cbar_intervals=None):
    """
    The second power test is more a performance test for  a fixed detection thresholds,
    plot different heatmaps each for a specific batch size
    the axes are the rhos and delta_rhos
    """
    for batch in full_stat_structure.keys():
        plot_pow_test_2_single_heat_map(full_stat_structure[batch], rhos, delta_rhos, batch, file_prefix,
                                        detection_threshold, cbar_name, cbar_intervals)


def plot_pow_test_2_single_heat_map(stat_matrix, rhos, delta_rhos, batch, file_prefix, detection_threshold,
                                    cbar_name, cbar_interval=None):
    fig, ax = plt.subplots()
    if cbar_interval:
        vmin = cbar_interval[0]
        vmax = cbar_interval[1]
    else:
        vmin = np.amin(stat_matrix)
        vmax = np.amax(stat_matrix)
    heat_map = ax.imshow(stat_matrix, cmap="PuOr", vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(heat_map, ax=ax)
    cbar.ax.set_ylabel(cbar_name, rotation=-90, va="bottom")
    x_labels = [str(i) for i in delta_rhos]
    y_labels = [str(i) for i in rhos]
    plt.xlabel("Intensity Ratio Rel. Change")
    plt.ylabel("Intensity Ratio")
    plt.xticks([i for i in range(len(delta_rhos))], x_labels)
    plt.yticks([i for i in range(len(rhos))], y_labels)
    plt.title("Batch size: {}, Detection threshold: {}, Serv. Rate=100".format(batch, detection_threshold))
    # Loop over data dimensions and create text annotations.
    for i in range(len(rhos)):
        for j in range(len(delta_rhos)):
            text = ax.text(j, i, "{0:2.2f}".format(stat_matrix[i, j]), ha="center", va="center", color="w")
            print(text)
    file_name = file_prefix + "_batch_{}.png".format(batch)
    plt.savefig(file_name, dpi=800)
    plt.show()


def plot_power_test_2_contours(full_stat_structure, batch_sizes, rhos, delta_rhos, file_prefix, detection_threshold):
    """
    The second power test is more a performance test for  a fixed detection thresholds,
    plot different heatmaps each for a specific batch size
    the axes are the rhos and delta_rhos
    """
    for batch in batch_sizes:
        plot_power_test_contour_2(full_stat_structure[batch], rhos, delta_rhos, batch, file_prefix,
                                  detection_threshold)


def plot_power_test_contour_2(power_mat, rhos, delta_rhos, batch, file_name, threshold):
    xx, yy = np.meshgrid(rhos, delta_rhos)
    plt.contour(xx, yy, power_mat)
    plt.colorbar()
    plt.xlabel('Intensity Ratios')
    plt.ylabel('Changes in Intensity Ratio')
    plt.title("Batch size: {}, Detection threshold: {}".format(batch, threshold))
    file_name = file_name + "_batch_{}.png".format(batch)
    plt.savefig(file_name, dpi=800)
    plt.show()


def plot_power_test_contour(power_mat, batch_size, thresholds, file_name, type=None):
    xx, yy = np.meshgrid(batch_size, thresholds)
    plt.contour(xx, yy, np.transpose(power_mat))
    plt.colorbar()
    plt.xlabel('Batch Size')
    plt.ylabel('Threshold')
    if type:
        plt.title(type)
    plt.savefig(file_name, dpi=800)
    plt.show()


def plot_power_test_surface(power_mat, batch_size, thresholds, file_name, type=None):
    xx, yy = np.meshgrid(batch_size, thresholds)
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, np.transpose(power_mat), cmap='viridis')
    plt.xlabel('Batch Size')
    plt.ylabel('Threshold')
    if type:
        plt.title(type)
    plt.savefig(file_name, dpi=800)
    plt.show()


def plot_3d_scatter(x_alpha, y_ht, z_td, batch_size, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_alpha, y_ht, z_td, marker='o')
    ax.set_xlabel('Significance level alpha')
    ax.set_ylabel('Detection threshold h_t')
    ax.set_zlabel('Detection delay t_d')
    ax.set_title("Batch Size= {}".format(batch_size))
    plt.savefig('./Figures/Power_Test/' + file_name)
    plt.show()


def plot_third_power_test(file_prefix, detection_delay, false_alarm_rate, threshold, batch):
    file_name = file_prefix + "_batch_{}_threshold.png".format(batch)
    title_str = "Threshold={}, Batch Size={}".format(threshold, batch)
    rho_relative_changes = []
    detection_delay_list = []
    false_alarm_rate_list = []
    for rho_change in detection_delay.keys():
        detection_delay_list.append(detection_delay[rho_change])
        false_alarm_rate_list.append(false_alarm_rate[rho_change])
        rho_relative_changes.append(rho_change)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    plt.semilogx(rho_relative_changes, detection_delay_list, '-.', color=color)
    ax1.set_xlabel("Relative Traffic Ratio Change")
    ax1.set_ylabel("Detection Delay", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('False Alarm Rate', color=color)
    ax2.semilogx(rho_relative_changes, false_alarm_rate_list, '-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title_str)
    plt.savefig(file_name)
    # plt.show()
    plt.close()


def plot_double_y_axis(y_1, y_2, x_axis, names, rho, file_prefix, held_parameter, held_value):
    """
    :param y_1: parameter on the left-axis
    :param y_2: parameter on the righht-axis
    :param x_axis: parameter on the x-axis
    :param names: left axis, right axis and x-axis names in that order
    :param file_prefix: prefix of the file
    :param rho: intensity ratio
    :param held_parameter: batch or delta_rho/rho what is being held constant
    :param held_value: value of batch or delta_rho/rho
    """
    file_name = file_prefix + "_{}_{}_rho_{}.png".format(held_parameter, held_value, rho)
    title_str = " {}={}".format(held_parameter, held_value)
    left_axis_name, right_axis_name, x_axis_name = names

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    plt.semilogx(x_axis, y_1, '-.', color=color)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(left_axis_name, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel(right_axis_name, color=color)
    ax2.plot(x_axis, y_2, '-', color=color)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()
    plt.title(title_str)
    plt.savefig(file_name, dpi=500)
    # plt.show()
    plt.close()
