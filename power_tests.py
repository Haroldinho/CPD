"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: power_tests.py
Description: Perform power tests using a Neyman-Pearson type rule to select the sequence of thresholds and batch sizes
required for a Generalized Likelihood Ratio Test (GLRT) on dependent wait times

Approach:
#1
Evaluation metrics: Detection delay (ARL1), false alarm rate probability (PFA),
time between two consecutive false detections (ARL0), ...

4/16/2020: Only using ladder times for now
Author: Harold Nemo Adodo Nikoue
part of the partial observability thesis
"""
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from batch_means_methods import create_nonoverlapping_batch_means
from estimation_facility import GLRTChangePointDetector, DetectionStatistics, CuSumChangePointDetector
from generate_m_m_1_processes import get_expected_wait_time, get_std_wait_time, simulate_ladder_point_process
from performance_statistics_facility import find_detection_index
from utilities import PowerTestLogger
from visualization import plot_power_test_contour, plot_power_test_2_heatmap, plot_power_test_contour_2, \
    plot_third_power_test, plot_double_y_axis


@dataclass(order=True)
class DetectionParam:
    detection_delay: float = field(default=0.0, compare=True)
    detection_threshold: float = field(default=0.0, compare=True)


def return_summary_detection_statistics(detection_param_list, percentile=None):
    # remove all large values
    detection_param_list = [val for val in detection_param_list if val.detection_delay < 1e9]
    detection_param_list = sorted(detection_param_list)
    standard_error = detection_param_list[-1].detection_delay - detection_param_list[0].detection_delay
    print(detection_param_list[0].detection_delay, detection_param_list[-1].detection_delay, standard_error)
    std_error_threshold = detection_param_list[-1].detection_threshold - detection_param_list[0].detection_threshold
    print("For this batch size, the standard error in detection delay is {0:2.4f}".format(standard_error), end='\t')
    print("The standard error in detection threshold is {0:2.4f}".format(std_error_threshold))
    if percentile:
        percentile_idx = int(percentile * len(detection_param_list))
        target_object = detection_param_list[percentile_idx]
        percentile_threshold = target_object.detection_threshold
        percentile_delay = target_object.detection_delay
        print("For a {0:2.0f} percentile, the threshold is {1:2.4f} and the detection delay is {2:2.4f}".format(
            percentile * 100.0, percentile_threshold, percentile_delay
        ))
    else:
        percentile = 0.05
        percentile_idx = int(percentile * len(detection_param_list))
        target_object = detection_param_list[percentile_idx]
        percentile_threshold = target_object.detection_threshold
        percentile_delay = target_object.detection_delay
        print("For a {0:2.0f} percentile, the threshold is {1:2.4f} and the detection delay is {2:2.4f}".format(
            percentile * 100.0, percentile_threshold, percentile_delay
        ))
        percentile = 0.02
        percentile_idx = int(percentile * len(detection_param_list))
        target_object = detection_param_list[percentile_idx]
        percentile_threshold = target_object.detection_threshold
        percentile_delay = target_object.detection_delay
        print("For a {0:2.0f} percentile, the threshold is {1:2.4f} and the detection delay is {2:2.4f}".format(
            percentile * 100.0, percentile_threshold, percentile_delay
        ))
        percentile = 0.01
        percentile_idx = int(percentile * len(detection_param_list))
        target_object = detection_param_list[percentile_idx]
        percentile_threshold = target_object.detection_threshold
        percentile_delay = target_object.detection_delay
        print("For a {0:2.0f} percentile, the threshold is {1:2.4f} and the detection delay is {2:2.4f}".format(
            percentile * 100.0, percentile_threshold, percentile_delay
        ))

    return percentile_threshold, percentile_delay


def power_test_training(arr_rate_pre, serv_rate_pre, arr_rate_post, serv_rate_post, desired_alpha):
    """
    Multiple tests to study the interplay between batch size, detection threshold, significance levels and
    detection delay
    It is used in a Neyman Pearson type test where the significance level is fixed and the best detection rate is sought
    I want to save:
        - detection threshold
        - false alarm rate
        - detection delay
        - missed detection probability/rate
        - batch size
    """
    num_experiments = 10000
    start_time = 0
    end_time = 1000
    my_arrival_rates = [arr_rate_pre, arr_rate_post]
    my_service_rates = [serv_rate_pre, serv_rate_post]
    thresholds = np.linspace(1e-2, 8)
    time_of_changes = [0, end_time / 2]
    batch_sizes = [20 + 5 * i for i in range(0, 10)]
    batch_threshold_dict = {batch: 0 for batch in batch_sizes}
    batch_delay_dict = {batch: 0 for batch in batch_sizes}
    log_directory = "./Results/"
    batch_threshold_log = PowerTestLogger(log_directory + "batch_threshold_log")
    batch_delay_log = PowerTestLogger(log_directory + "batch_delay_log")
    detection_delay_mat = np.zeros((len(batch_sizes), len(thresholds)))
    missed_detection_mat = np.zeros((len(batch_sizes), len(thresholds)))
    missed_detection_by_threshold_by_batch_size = {batch: {} for batch in batch_sizes}
    for batch_idx, batch_size in enumerate(batch_sizes):
        # Draw some wait times using the distribution
        # significance levels
        x_alphas = []
        y_h_t = []
        z_t_d = []
        detection_param_list = []
        missed_detection_by_threshold = {i: 0 for i in range(len(thresholds))}
        for exp_num in range(num_experiments):
            wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
                                                                      time_of_changes, my_service_rates)

            for threshold_idx, threshold in enumerate(thresholds):
                batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                         batch_size=batch_size)
                # 3. Run a change point detection algorithm on the m
                glrt_change_point_estimator = GLRTChangePointDetector(threshold, None, dist_type='gaussian')
                glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                deds_glrt_detection_statistics = DetectionStatistics(glrt_change_point_estimator.detection_result,
                                                                     batch_centers)
                glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
                detection_index = find_detection_index(glrt_detected_change_times, time_of_changes[1])
                if 0 <= detection_index < len(glrt_detected_change_times):
                    detection_delay = glrt_detected_change_times[detection_index] - time_of_changes[1]
                    assert (detection_delay >= 0)
                    false_detection_list = glrt_detected_change_times[:detection_index] \
                                           + glrt_detected_change_times[detection_index + 1:]
                else:
                    missed_detection_by_threshold[threshold_idx] += 1
                    detection_delay = float('inf')
                    false_detection_list = glrt_detected_change_times
                false_detection_prob = len(false_detection_list) / float(len(batch_mean_wait_times))
                if false_detection_prob < desired_alpha:
                    x_alphas.append(false_detection_prob)
                    y_h_t.append(threshold)
                    z_t_d.append(detection_delay)
                    new_parameter = DetectionParam(detection_delay=detection_delay, detection_threshold=threshold)
                    detection_param_list.append(new_parameter)
                    detection_delay_mat[batch_idx][threshold_idx] = detection_delay

                # power_structure_by_batch[batch_size].insert(false_detection_prob, threshold, detection_delay)
        # plot_3d_scatter(x_alphas, y_h_t, z_t_d, batch_size, "scatter_plot_batch_{}.png".format(batch_size))
        batch_threshold, batch_delay = return_summary_detection_statistics(detection_param_list)
        batch_threshold_dict[batch_size] = batch_threshold
        batch_delay_dict[batch_size] = batch_delay
        for h_t, missed_detection_count in missed_detection_by_threshold.items():
            missed_detection_mat[batch_idx][h_t] = float(missed_detection_count) / num_experiments
        batch_threshold_log.write_data(batch_threshold_dict)
        batch_delay_log.write_data(batch_delay_dict)
        # generate statistics for the detection delay at a certain level alpha
        # Mostly care about the lowest detection delay for a significance level alpha
        # Focus now only on current batch size and significance level 0.05
        # main_structure = power_structure_by_batch[batch_size].return_matrix()
        # plot_power_test_contour(main_structure, alphas, batch_size, "contour_batch_{}.png".format(batch_size))
    plot_power_test_contour(detection_delay_mat, batch_sizes, thresholds,
                            "Figures/Power_Test/detection_delay_parametric_study_test4.png", "Detection Delay")
    plot_power_test_contour(missed_detection_mat, batch_sizes, thresholds,
                            "Figures/Power_Test/missed_detection_parametric_study_test4.png", "Missed Detection")
    print("Dictionary of threshold indexed by batch size")
    print(batch_threshold_dict)
    print("Dictionary of delays indexed by batch size")
    print(batch_delay_dict)


def main_change_mu(test='test-Four'):
    # 1. Set nominal parameters for the power test
    if test == 'test-One':
        # going from under utilized to over utilized
        lambda_nom = 4
        mu_nom = 10
        lambda_1 = 12
        mu_1 = mu_nom
    elif test == 'test-Two':
        # going from under-utilized to under-utilitzed
        lambda_nom = 4
        mu_nom = 10
        lambda_1 = 8
        mu_1 = mu_nom
    elif test == 'test-Three':
        # going from over-utilized to under-untilized
        lambda_nom = 12
        mu_nom = 10
        lambda_1 = 8
        mu_1 = mu_nom
    elif test == 'test-Four':
        # going from over-utilized to over-utilized
        lambda_nom = 11
        mu_nom = 10
        lambda_1 = 16
        mu_1 = mu_nom
    power_test_training(lambda_nom, mu_nom, lambda_1, mu_1, desired_alpha=0.05)


def run_second_power_test():
    """
    For a fixed detection threshold between
    For different intensity ratios,

    """
    # intensity ratio = \frac{\lambda}{\mu}
    rho_vec = [0.5, 0.8, 0.9]
    batch_sizes = [25 + 25 * i for i in range(0, 6)]
    mu = 100
    # relative change in intensity ratio
    delta_rho_rel = [0.01, 0.1, 1, 10, 100]
    desired_alpha = 0.05

    num_runs = 500
    start_time = 0
    end_time = 500
    detection_threshold = [0.8, 1.2, 2]
    threshold = detection_threshold[0]
    time_of_changes = [0, end_time / 2]

    log_directory = "./Results/"
    my_service_rates = [mu, mu]
    detection_delay_struct = {b: np.zeros((len(rho_vec), len(delta_rho_rel))) for b in batch_sizes}
    missed_detection_struct = {b: np.zeros((len(rho_vec), len(delta_rho_rel))) for b in batch_sizes}
    batch_delay_log = PowerTestLogger(log_directory + "batch_detection_delay_test_2_log")
    batch_missed_detection_log = PowerTestLogger(log_directory + "batch_missed_detection_test_2_log")
    for rho_idx, rho in enumerate(rho_vec):
        arr_rate_0 = mu * rho
        for delta_rho_idx, delta_rho in enumerate(delta_rho_rel):
            print(" Working on (rho={}, delta_rho={})".format(rho, delta_rho))
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            detection_delay_dict = defaultdict(list)
            missed_detection_dict = defaultdict(int)
            for run_idx in range(num_runs):
                wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
                                                                          time_of_changes, my_service_rates)
                for batch_size in batch_sizes:
                    batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                             batch_size=batch_size)
                    # 3. Run a change point detection algorithm on the m
                    glrt_change_point_estimator = GLRTChangePointDetector(threshold, None, dist_type='gaussian')
                    glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                    deds_glrt_detection_statistics = DetectionStatistics(glrt_change_point_estimator.detection_result,
                                                                         batch_centers)
                    glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
                    detection_index = find_detection_index(glrt_detected_change_times, time_of_changes[1])
                    if 0 <= detection_index < len(glrt_detected_change_times):
                        detection_delay = glrt_detected_change_times[detection_index] - time_of_changes[1]
                        assert (detection_delay >= 0)
                        false_detection_list = glrt_detected_change_times[:detection_index] \
                                               + glrt_detected_change_times[detection_index + 1:]
                    else:  # nothing was detected
                        detection_delay = np.nan  # change that to float('inf')
                        false_detection_list = glrt_detected_change_times
                    false_detection_prob = len(false_detection_list) / float(len(batch_mean_wait_times))
                    if false_detection_prob < desired_alpha:
                        if np.isnan(detection_delay):
                            missed_detection_dict[batch_size] += 1
                        else:
                            detection_delay_dict[batch_size].append(detection_delay)
            for batch in batch_sizes:
                if len(detection_delay_dict[batch]) > 0:
                    detection_delay_struct[batch][rho_idx, delta_rho_idx] = np.mean(detection_delay_dict[batch])
                else:
                    detection_delay_struct[batch][rho_idx, delta_rho_idx] = np.nan
                missed_detection_struct[batch][rho_idx, delta_rho_idx] = missed_detection_dict[batch] / float(num_runs)
                batch_delay_log.write_data(detection_delay_struct)
                batch_missed_detection_log.write_data(missed_detection_struct)
    detection_delay_file_prefix = "./Figures/Power_Test/Detection_Delay_Test2_Heatmap"
    missed_detection_file_prefix = "./Figures/Power_Test/Missed_Detection_Test2_Heatmap"
    detection_delay_contour_file_prefix = "./Figures/Power_Test/Detection_Delay_Test2_Contour"
    missed_detection_contour_file_prefix = "./Figures/Power_Test/Missed_Detection_Test2_Contour"
    plot_power_test_2_heatmap(detection_delay_struct, batch_sizes, rho_vec, delta_rho_rel,
                              detection_delay_file_prefix, threshold, "Detection Delay")
    #    plot_power_test_2_heatmap(missed_detection_struct, batch_sizes, rho_vec, delta_rho_rel,
    #                              missed_detection_file_prefix, threshold, "Missed Detection", [0, 1])
    plot_power_test_contour_2(detection_delay_struct, batch_sizes, rho_vec, delta_rho_rel,
                              detection_delay_contour_file_prefix, threshold)


#    plot_power_test_contour_2(missed_detection_struct, batch_sizes, rho_vec, delta_rho_rel,
#                              missed_detection_contour_file_prefix, threshold, "Missed Detection", [0, 1])

def run_third_power_test():
    """
    May 4th
    Finding the optimal batch size using 2D plots
    New modification: Look at median detection delay, set  detection delay to infinity if there is no correct detection
    Missed detections and False alarm rates are unaffected.
    """
    # Fix rho, detection thresholds, batch sizes
    rho = 0.8
    batch_sizes = [25 + 25 * i for i in range(0, 6)]
    mu = 100
    # relative change in intensity ratio
    delta_rho_rel = [0.01, 0.1, 1, 10, 100]
    detection_thresholds = [0.01, 0.1, 0.8, 1.2]

    num_runs = 1000
    start_time = 0
    end_time = 500
    time_of_changes = [0, end_time / 2]

    my_service_rates = [mu, mu]
    arr_rate_0 = mu * rho
    log_directory = "./Results/"
    batch_delay_log = PowerTestLogger(log_directory + "batch_detection_delay_test_3_log")
    batch_missed_detection_log = PowerTestLogger(log_directory + "batch_missed_detection_test_3_log")
    batch_false_alarm_log = PowerTestLogger(log_directory + "batch_false_alarm_test_3_log")

    for threshold_idx, threshold in enumerate(detection_thresholds):
        detection_delay_dict_dict = {b: {delta_rho: 0 for delta_rho in delta_rho_rel} for b in batch_sizes}
        false_alarm_dict_dict = {b: {delta_rho: 0 for delta_rho in delta_rho_rel} for b in batch_sizes}
        missed_detection_dict_dict = {b: {delta_rho: 0 for delta_rho in delta_rho_rel} for b in batch_sizes}
        for delta_rho_idx, delta_rho in enumerate(delta_rho_rel):
            print(" Working on (rho={}, delta_rho={}, detection_threshold={})".format(rho, delta_rho,
                                                                                      threshold))
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            detection_delay_dict = defaultdict(list)
            missed_detection_dict = defaultdict(int)
            false_alarm_dict = defaultdict(list)
            for run_idx in range(num_runs):
                wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
                                                                          time_of_changes, my_service_rates)
                for batch_size in batch_sizes:
                    if run_idx == 0:
                        batch_delay = batch_size * 1.0 / my_arrival_rates[1]
                        print("First run with batch size: {} and avg. batch delay: {}".format(batch_size, batch_delay))
                    batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                             batch_size=batch_size)
                    # 3. Run a change point detection algorithm on the m
                    glrt_change_point_estimator = GLRTChangePointDetector(threshold, None, dist_type='gaussian')
                    glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                    deds_glrt_detection_statistics = DetectionStatistics(glrt_change_point_estimator.detection_result,
                                                                         batch_centers)
                    glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
                    detection_index = find_detection_index(glrt_detected_change_times, time_of_changes[1])
                    if 0 <= detection_index < len(glrt_detected_change_times):
                        detection_delay = glrt_detected_change_times[detection_index] - time_of_changes[1]
                        assert (detection_delay >= 0)
                        false_detection_list = glrt_detected_change_times[:detection_index] \
                                               + glrt_detected_change_times[detection_index + 1:]
                    else:  # nothing was detected
                        detection_delay = float('inf')
                        false_detection_list = glrt_detected_change_times
                    false_detection_prob = len(false_detection_list) / float(len(batch_mean_wait_times))
                    false_alarm_dict[batch_size].append(false_detection_prob)
                    detection_delay_dict[batch_size].append(detection_delay)
                    if detection_index >= len(glrt_detected_change_times):
                        missed_detection_dict[batch_size] += 1
            for batch in batch_sizes:
                detection_delay_dict_dict[batch][delta_rho] = np.median(detection_delay_dict[batch])
                false_alarm_dict_dict[batch][delta_rho] = np.mean(false_alarm_dict[batch])
                missed_detection_dict_dict[batch][delta_rho] = missed_detection_dict[batch] / float(num_runs)

            batch_false_alarm_log.write_data(false_alarm_dict_dict)
            batch_delay_log.write_data(detection_delay_dict_dict)
            batch_missed_detection_log.write_data(missed_detection_dict_dict)

        combined_plot_file_prefix = "./Figures/Power_Test3/Power_Test3_2D_plot_{}".format(threshold_idx)
        for batch in batch_sizes:
            plot_third_power_test(combined_plot_file_prefix, detection_delay_dict_dict[batch],
                                  false_alarm_dict_dict[batch], threshold, batch)


def run_fourth_power_test(rho, delta_rho, batch_sizes, detection_thresholds, num_runs, test_type="GLRT"):
    start_time = 0
    end_time = 500
    time_of_changes = [0, end_time / 2]
    mu = 10
    my_service_rates = [mu, mu]
    arr_rate_0 = mu * rho
    arr_rate_1 = arr_rate_0 * (1 + delta_rho)
    my_arrival_rates = [arr_rate_0, arr_rate_1]
    # Expected queueing time p. 343 of Gallager
    mean_wait_time_0 = get_expected_wait_time(arr_rate_0, mu)
    mean_wait_time_1 = get_expected_wait_time(arr_rate_1, mu)
    std_wait_time_0 = get_std_wait_time(arr_rate_0, mu)
    std_wait_time_1 = get_std_wait_time(arr_rate_1, mu)
    print("Running sim for rho={0:2.4f}, delta_rho={1:2.4f}. ".format(rho, delta_rho))
    batch_false_alarm_dict = defaultdict(list)
    batch_detection_delay_dict = defaultdict(list)
    batch_detection_threshold_dict = defaultdict(list)
    batch_missed_detection_dict = {threshold: defaultdict(int) for threshold in detection_thresholds}

    result_false_alarm_dict = dict()
    result_detection_delay_dict = dict()
    result_detection_threshold_dict = dict()
    result_missed_detection_dict = dict()
    for _ in range(num_runs):
        wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
                                                                  time_of_changes, my_service_rates)
        threshold_by_batch_size = {b: 0 for b in batch_sizes}
        for batch_size in batch_sizes:
            batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                     batch_size=batch_size)
            #             print("{} batch centers".format(len(batch_centers)))
            detection_delays_by_threshold = []
            false_alarm_rate_by_threshold = []

            for threshold in detection_thresholds:
                # 3. Run a change point detection algorithm on the m
                if test_type == "GLRT":
                    glrt_change_point_estimator = GLRTChangePointDetector(threshold, None, dist_type='gaussian')
                    glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                    deds_glrt_detection_statistics = DetectionStatistics(glrt_change_point_estimator.detection_result,
                                                                         batch_centers)
                    # list of all detected change times
                    detected_change_times = deds_glrt_detection_statistics.change_point_times
                elif test_type == "CuSum":
                    cusum_change_point_estimator = CuSumChangePointDetector((mean_wait_time_0, std_wait_time_0),
                                                                            (mean_wait_time_1, std_wait_time_1),
                                                                            threshold, dist_type='normal')
                    #                     cusum_change_point_estimator = CuSumChangePointDetector((arr_rate_0, mu),
                    #                                                                             (arr_rate_1, mu),
                    #                                                                             threshold, dist_type='normal2')
                    cusum_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                    cusum_detection_statistics = DetectionStatistics(cusum_change_point_estimator.detection_results,
                                                                     batch_centers)
                    detected_change_times = cusum_detection_statistics.change_point_times

                detection_index = find_detection_index(detected_change_times, time_of_changes[1])
                if 0 <= detection_index < len(detected_change_times):
                    detection_delay = detected_change_times[detection_index] - time_of_changes[1]
                    assert (detection_delay >= 0)
                    false_detection_list = detected_change_times[:detection_index] \
                                           + detected_change_times[detection_index + 1:]
                else:  # nothing was detected
                    detection_delay = float('inf')
                    false_detection_list = detected_change_times
                    batch_missed_detection_dict[threshold][batch_size] += 1
                # Average Run Length to False detection
                arl_0 = np.mean(np.diff(false_detection_list))
                false_detection_prob = len(false_detection_list) / float(len(batch_mean_wait_times))
                false_alarm_rate_by_threshold.append(false_detection_prob)
                detection_delays_by_threshold.append(detection_delay)
            # END for threshold
            # pick the index of the threshold that minimizes the detection delay
            idx_delay = np.argmin(detection_delays_by_threshold)
            batch_detection_delay_dict[batch_size].append(detection_delays_by_threshold[idx_delay])
            batch_detection_threshold_dict[batch_size].append(detection_thresholds[idx_delay])
            batch_false_alarm_dict[batch_size].append(false_alarm_rate_by_threshold[idx_delay])
        # END BATCH_SIZE
        # For each batch, I want to return the median detection delay,
        # the corresponding detection threshld and false alarm rate

    # END RUNS
    # find median idx
    for batch_size in batch_sizes:
        med_idx = batch_detection_delay_dict[batch_size].index(np.percentile(batch_detection_delay_dict[batch_size], 50,
                                                                             interpolation='nearest'))
        result_detection_delay_dict[batch_size] = batch_detection_delay_dict[batch_size][med_idx]
        result_detection_threshold_dict[batch_size] = batch_detection_threshold_dict[batch_size][med_idx]
        selected_threshold = batch_detection_threshold_dict[batch_size][med_idx]
        result_false_alarm_dict[batch_size] = batch_false_alarm_dict[batch_size][med_idx]
        missed_detect = batch_missed_detection_dict[selected_threshold][batch_size]
        result_missed_detection_dict[batch_size] = missed_detect / float(num_runs)
    return result_detection_delay_dict, result_detection_threshold_dict, result_false_alarm_dict, \
           result_missed_detection_dict


def log_data(log_directory, rho, delta_rho_rel, detection_delay, false_alarm, missed_detection, detection_threshold):
    detection_delay_log = PowerTestLogger(log_directory + "expanded_test_delay_{}_{}".format(rho,
                                                                                             delta_rho_rel))
    false_alarm_log = PowerTestLogger(log_directory + "expanded_false_alarm_{}_{}".format(rho,
                                                                                          delta_rho_rel))
    missed_detection_log = PowerTestLogger(log_directory + "expanded_missed_detection_{}_{}".format(
        rho, delta_rho_rel))
    detection_threshold_log = PowerTestLogger(log_directory + "expanded_detection_threshold_{}_{}".format(
        rho, delta_rho_rel))
    detection_delay_log.write_data(detection_delay)
    false_alarm_log.write_data(false_alarm)
    missed_detection_log.write_data(missed_detection)
    detection_threshold_log.write_data(detection_threshold)


def run_expanded_power_test(test_type="GLRT"):
    """
        Run multidimensional test to decide batch size and detection threshold required for different intensity ratio
        changes
        plot detection delay and false alarm rate vs change in intensity ratio or batch size
        :param test_type:  parameter to decide whether to run Generalized Likelihood Ratio Test or CuSum
    """
    batch_sizes = [5, 10, 20, 25, 35, 50, 60, 75, 80, 100, 125, 150]
    detection_thresholds = [0.01, 0.1, 0.5, 0.8, 1, 1.2, 1.4, 1.6, 2.0, 2.5]
    rho_vec = [0.5, 0.8, 0.9]
    delta_rho_rel_vec = [0.01, 0.1, 1, 10, 100]
    num_runs = 10000
    detection_delay_by_delta_rho = {batch: defaultdict() for batch in batch_sizes}
    detection_threshold_by_delta_rho = {batch: defaultdict() for batch in batch_sizes}
    false_alarm_by_delta_rho = {batch: defaultdict() for batch in batch_sizes}
    missed_detection_by_delta_rho = {batch: defaultdict() for batch in batch_sizes}

    log_directory = "./Results/"
    for rho in rho_vec:
        for delta_rho_rel in delta_rho_rel_vec:
            detection_delay_dict, detection_threshold_dict, false_alarm_dict, missed_detection_dict = \
                run_fourth_power_test(rho, delta_rho_rel, batch_sizes, detection_thresholds, num_runs, test_type)

            log_data(log_directory, rho, delta_rho_rel, detection_delay_dict, false_alarm_dict, missed_detection_dict,
                     detection_threshold_dict)  # logging data
            for batch in batch_sizes:
                detection_delay_by_delta_rho[batch][delta_rho_rel] = detection_delay_dict[batch]
                detection_threshold_by_delta_rho[batch][delta_rho_rel] = detection_threshold_dict[batch]
                missed_detection_by_delta_rho[batch][delta_rho_rel] = missed_detection_dict[batch]
                false_alarm_by_delta_rho[batch][delta_rho_rel] = false_alarm_dict[batch]
                # Plot all three statistics vs. delta_rho_rel for different batch_sizes
            # END BATCH
        # END Delta_rho
        print("Plotting")
        for batch in batch_sizes:
            combined_plot_1 = "./Figures/Power_Test4/Power_Test4_{}_batch_{}_rho_{}_type1".format(test_type, batch, rho)
            print(len(list(detection_delay_by_delta_rho[batch].values())),
                  len(list(false_alarm_by_delta_rho[batch].values())),
                  len(delta_rho_rel_vec))
            plot_double_y_axis(y_1=list(detection_delay_by_delta_rho[batch].values()),
                               y_2=list(false_alarm_by_delta_rho[batch].values()),
                               x_axis=delta_rho_rel_vec, names=["Detection_Delay", "FAR", "DeltaRho_rel"], rho=rho,
                               file_prefix=combined_plot_1, held_parameter="Batch_Size", held_value=batch)

            combined_plot_2 = "./Figures/Power_Test4/Power_Test4_{}_batch_{}_rho_{}_type2".format(test_type, batch, rho)
            plot_double_y_axis(y_1=list(detection_delay_by_delta_rho[batch].values()),
                               y_2=list(detection_threshold_by_delta_rho[batch].values()),
                               x_axis=delta_rho_rel_vec, names=["Detection_Delay", "Detection_Threshold",
                                                                "DeltaRho_rel"], rho=rho,
                               file_prefix=combined_plot_2, held_parameter="Batch_Size", held_value=batch)

            combined_plot_3 = "./Figures/Power_Test4/Power_Test4_{}_batch_{}_rho_{}_type3".format(test_type, batch, rho)
            plot_double_y_axis(y_1=list(detection_delay_by_delta_rho[batch].values()),
                               y_2=list(missed_detection_by_delta_rho[batch].values()),
                               x_axis=delta_rho_rel_vec, names=["Detection_Delay", "Missed_Detection_Rate",
                                                                "DeltaRho_rel"], rho=rho,
                               file_prefix=combined_plot_3, held_parameter="Batch_Size", held_value=batch)

            combined_plot_4 = "./Figures/Power_Test4/Power_Test4_{}_batch_{}_rho_{}_type4".format(test_type, batch, rho)
            plot_double_y_axis(y_1=list(false_alarm_by_delta_rho[batch].values()),
                               y_2=list(missed_detection_by_delta_rho[batch].values()),
                               x_axis=delta_rho_rel_vec, names=["FAR", "Missed_Detection_Rate",
                                                                "DeltaRho_rel"], rho=rho,
                               file_prefix=combined_plot_4, held_parameter="Batch_Size", held_value=batch)
            combined_plot_5 = "./Figures/Power_Test4/Power_Test4_{}_batch_{}_rho_{}_type2".format(test_type, batch, rho)
            plot_double_y_axis(y_1=list(false_alarm_by_delta_rho[batch].values()),
                               y_2=list(detection_threshold_by_delta_rho[batch].values()),
                               x_axis=delta_rho_rel_vec, names=["FAR", "Detection_Threshold",
                                                                "DeltaRho_rel"], rho=rho,
                               file_prefix=combined_plot_5, held_parameter="Batch_Size", held_value=batch)


if __name__ == "__main__":
    """
        Decision parameters:
            - Batch size
            - Detection Threshold
            - False alarm Rate / Significance Level / ARL_0
            - Lambda
            - Mu
            - Nu : Location of change point
            - Change in intensity ratio (holding mu constant)
        Performance Metrics:
            - False Alarm Rate/ ARL_0 / Significance level 
            - Detection  Delay / ARL_1 / Power
            - Missed Detection Probability (of not detecting the change)
    """
    # main_change_mu()
    run_expanded_power_test(test_type='CuSum')
