"""
Simulate the detection delays by using the already computed and stored thresholds for different ARL_0 (They were
computed in simulate_arl_0.py).

"""
import itertools
import math
import random
from collections import defaultdict
from enum import Enum, unique, auto
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.rinterface import R_VERSION_BUILD
from rpy2.robjects import FloatVector, r
from rpy2.robjects.packages import STAP
from rpy2.robjects.packages import importr

from average_run_length_loader import ThresholdSelector
from batch_means_methods import create_nonoverlapping_batch_means
from batch_selection import BatchSelectorOnAutocorrelation
from estimation_facility import GLRTChangePointDetector, DetectionStatistics
from generate_m_m_1_processes import simulate_ladder_point_process, simulate_deds_return_wait_times, \
    simulate_deds_return_age, simulate_deds_return_queue_length, simulate_deds_return_age_queue_wait
# from plot_output_performance_curve import plot_density_histogram_of_detections
from utilities import PowerTestLogger


@unique
class RuntimeType(Enum):
    SPECIFIED = auto()
    RELATIVE = auto()
    LIST = auto()


@unique
class ChangePointLocationChoice(Enum):
    SPECIFIED = auto()
    UNIFORM = auto()
    RELATIVE = auto()
    GEOMETRIC = auto()


class SimulationSetting:
    def __init__(self):
        """
        Define the defaults settings for the simulation

        """
        # SPECIFIED: The length of one run is a specific number given by the user for all runs and all scenarios
        # RELATIVE: The run length is a ratio of the given ARL_0
        self.runtime_type = RuntimeType.SPECIFIED
        self.runtime_num = -1
        # SPECIFIED: A specific location of the change point is given
        # UNIFORM: The change point location is taken uniformly at random from t=5 to t_f - 2 where t_f is the
        # end of the sim
        # RELATIVE: The change point location is taken to be a fraction of the total run time
        self.change_point_location_type = ChangePointLocationChoice.SPECIFIED
        self.change_point_location = None
        # Flag to decide whether the correct detection probabilities should be fixed at fixed run times
        # (detection times)
        self.use_detection_window = False

    def set_runtime(self, input_num):
        """
        Set the run time length
        or set the run time ratio of ARL_0
        """
        if self.runtime_type == RuntimeType.RELATIVE:
            assert (0 <= input_num <= 1)
        self.runtime_num = input_num

    def set_change_point(self, change_point_val):
        """
        Set the change point location
        """
        assert (change_point_val > 0)
        if self.change_point_location_type == ChangePointLocationChoice.RELATIVE:
            assert (change_point_val <= 1)
            self.change_point_location = change_point_val * self.runtime_num
        elif self.change_point_location_type == ChangePointLocationChoice.SPECIFIED:
            self.change_point_location = change_point_val


def generate_random_change_point_time(arl_0, end_of_warm_up_time):
    """
    Pick a random change point time uniformly at random
    """
    event_time = float("inf")
    u = random.random()
    if u > np.exp(-1):
        # generate a random numbr from 1 to arl_0 exclusive.
        event_time = random.randint(1, arl_0)
        while end_of_warm_up_time > event_time:
            event_time = random.randint(1, arl_0)
    return event_time


def disregard_by_num_of_samples_fraction(wait_times, dep_times, disregard_frac=0.2):
    # Delete a fixed fraction of the sameples
    n = len(wait_times)
    n_disregard = int(disregard_frac * n)
    return wait_times[n_disregard:], dep_times[n_disregard:]


def disregard_by_length_of_interval(wait_times, dep_times, end_time, disregard_frac=0.2):
    # time after which we start using/ collecting the samples
    effective_sample_time = disregard_frac * end_time
    return_wait_times = []
    return_dep_times = []
    for idx, dep_time in enumerate(dep_times):
        if dep_time > effective_sample_time:
            break
        else:
            return_dep_times.append(dep_time)
            return_wait_times.append(wait_times[idx])
    return wait_times, dep_times


def simulate_changepoints_in_waitingtimes_using_r_cpm(data_df: pd.DataFrame, rho: List[float], delta_rhos: List[float],
                                                      arr_rate_0: float, num_runs: int, start_time: float,
                                                      end_time: float,
                                                      my_service_rates: List[float],
                                                      batch_size: List[int], power_delay_log: float, cpm_func):
    """
    this code is to use the implementation of cpm in r directly from r using rpy2
    :param data_df: dataframe that will contain the performance characteristics of the test
    :param rho: list of intensity ratios
    :param delta_rhos: list of changes in intensity ratio
    :param arr_rate_0: initial arrival rate
    :param num_runs: number of runs
    :param start_time: start time of the sim 0 by default
    :param end_time: end time of the sim
    :param my_service_rates:
    :param my_thresholds:
    :param batch_size:
    :param cpm_func: r wrapper function to cpm
    :param power_delay_log: used to save data in between runs
    :return: dataframe of the data_df
    """
    true_positive = {delta_rho: 0 for delta_rho in delta_rhos}
    num_detection_dict = {delta_rho: 0 for delta_rho in delta_rhos}
    false_positive = {delta_rho: 0 for delta_rho in delta_rhos}
    false_negative = {delta_rho: 0 for delta_rho in delta_rhos}
    true_negative = {delta_rho: 0 for delta_rho in delta_rhos}
    detection_delay_dict = {delta_rho: [] for delta_rho in delta_rhos}
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    # distribution of time till false detection
    # just keep a list of all the detection times when you have a false positive
    false_positive_list = []
    for run_idx in range(num_runs):
        print(f"run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            if delta_rho == 0.0:
                time_of_change = float('inf')
            else:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]

            orig_wait_times, orig_wait_times_ts = simulate_deds_return_wait_times(start_time, end_time,
                                                                                  my_arrival_rates, time_of_changes,
                                                                                  my_service_rates)
            # create an artificial warm-up period
            wait_times, wait_times_ts = disregard_by_length_of_interval(orig_wait_times, orig_wait_times_ts, end_time,
                                                                        disregard_frac)
            # print(f"batch_mean_wait_times is of type {type(batch_mean_wait_times)} and length {len(batch_mean_wait_times)}")
            if batch_size:
                batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                         batch_size=batch_size)
                rbatch_mean_wait_times = FloatVector(batch_mean_wait_times)
                r.assign('remote_batch_mean_wait_times', rbatch_mean_wait_times)
                r_estimated_changepoint_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_wait_times)
            else:
                r_estimated_changepoint_index = cpm_func.Detect_r_cpm_GaussianChangePoint(wait_times)

            estimated_changepoint_idx = r_estimated_changepoint_index[0]
            # print(f"cpm returned {r_estimated_changepoint_index} which should be {estimated_changepoint_idx}")

            if np.isinf(time_of_change):
                if estimated_changepoint_idx > 0:
                    num_detection_dict[delta_rho] += 1
                    false_positive[delta_rho] += 1
                    estimated_changepoint_idx -= 1  # because the index starts at one
                    if batch_size:
                        try:
                            estimated_changepoint = batch_centers[estimated_changepoint_idx]
                        except IndexError:
                            print(
                                f"estimated changepoint idx {estimated_changepoint_idx} vs. array of length {len(batch_centers)}")
                            raise
                    else:
                        estimated_changepoint = wait_times_ts[estimated_changepoint_idx]
                    false_positive_list.append(estimated_changepoint)
                else:
                    true_negative[delta_rho] += 1
            else:
                if estimated_changepoint_idx > 0:
                    estimated_changepoint_idx -= 1
                    num_detection_dict[delta_rho] += 1
                    if batch_size:
                        try:
                            estimated_changepoint = batch_centers[estimated_changepoint_idx]
                        except IndexError:
                            print(
                                f"estimated changepoint idx {estimated_changepoint_idx} vs. array of length {len(batch_centers)}")
                    else:
                        estimated_changepoint = wait_times_ts[estimated_changepoint_idx]
                    print(f"estimated change point location: {estimated_changepoint} vs actual {time_of_change} ")
                    detection_delay = estimated_changepoint - time_of_change
                    if detection_delay >= 0:
                        true_positive[delta_rho] += 1
                        detection_delay_dict[delta_rho].append(detection_delay)
                    else:
                        false_positive[delta_rho] += 1
                        false_positive_list.append(estimated_changepoint)

                else:
                    false_negative[delta_rho] += 1
    for delta_rho in delta_rhos:
        denominator_prec = (true_positive[delta_rho] + false_positive[delta_rho])
        denominator_tp = (false_negative[delta_rho] + true_positive[delta_rho])
        denominator_fp = (false_positive[delta_rho] + true_negative[delta_rho])
        if denominator_fp:
            fp_rate = false_positive[delta_rho] / float(denominator_fp)
        else:
            fp_rate = np.nan
        if denominator_tp > 0:
            missed_detection_prob = false_negative[delta_rho] / float(denominator_tp)
            tp_rate = true_positive[delta_rho] / float(denominator_tp)
        else:
            tp_rate = np.nan
            missed_detection_prob = np.nan
        sensitivity = tp_rate
        precision = true_positive[delta_rho] / float(denominator_prec) if denominator_prec > 0 else np.nan
        prob_detect = (true_positive[delta_rho] + false_positive[delta_rho]) / float(num_runs)
        if detection_delay_dict[delta_rho]:
            mean_detection_delay = np.mean(detection_delay_dict[delta_rho])
        else:
            mean_detection_delay = np.nan
        if batch_size:
            values_to_add = {
                'batch size': batch_size,
                'rho': rho, 'delta_rho': delta_rho,
                'arl_1': mean_detection_delay,
                'missed detection prob': missed_detection_prob,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                'correct_detection': prob_detect,
                'run length': end_time,
                "tp": true_positive[delta_rho],
                "fp": false_positive[delta_rho],
                "fn": false_negative[delta_rho],
                "tn": true_negative[delta_rho],
                "number detections": num_detection_dict[delta_rho],
                "recall": sensitivity,
                "precision": precision,
            }
        else:
            values_to_add = {
                'batch size': batch_size,
                'rho': rho, 'delta_rho': delta_rho,
                'arl_1': mean_detection_delay,
                'missed detection prob': missed_detection_prob,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                'correct_detection': prob_detect,
                'run length': end_time,
                "tp": true_positive[delta_rho],
                "fp": false_positive[delta_rho],
                "fn": false_negative[delta_rho],
                "tn": true_negative[delta_rho],
                "number detections": num_detection_dict[delta_rho],
                "recall": sensitivity,
                "precision": precision,
            }

        row_to_add = pd.Series(values_to_add)
        print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
        power_delay_log.write_data(data_df)
    return data_df, false_positive_list


def simulate_joint_hypothesis_outcome_conditioned_on_change(data_df: pd.DataFrame, rho: List[float],
                                                            delta_rhos: List[float], arr_rate_0: float, num_runs: int,
                                                            start_time: float, end_time: float,
                                                            my_service_rates: List[float], batch_size: List[int],
                                                            power_delay_log: float, cpm_func):
    """
    This code is to use the implementation of CPM in R directly from R using rpy2
    :param data_df: dataframe that will contain the performance characteristics of the test
    :param rho: list of intensity ratios
    :param delta_rhos: list of changes in intensity ratio
    :param arr_rate_0: initial arrival rate
    :param num_runs: number of runs
    :param start_time: start time of the sim 0 by default
    :param end_time: end time of the sim
    :param my_service_rates:
    :param batch_size:
    :param cpm_func: R wrapper function to cpm
    :param power_delay_log: used to save data in between runs
    :return: dataframe of the data_df
    """
    # Look at the age, queue, wait outcome given a change

    changepoint_age_pos_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}

    no_changepoint_age_pos_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_pos_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_pos_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_pos_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    # Distribution of time till false detection
    # just keep a list of all the detection times when you have a false positive
    num_changes = 0
    num_no_changes = 0
    for run_idx in range(num_runs):
        print(f"Run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:

            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            if delta_rho == 0.0:
                time_of_change = float('inf')
            else:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]
            queue_lengths, queue_length_times, mean_age_times, recording_times, wait_times, departure_times = \
                simulate_deds_return_age_queue_wait(start_time, end_time, my_arrival_rates,
                                                    time_of_changes, my_service_rates)

            age_of_customers, age_times_ts = disregard_by_length_of_interval(mean_age_times, recording_times, end_time,
                                                                             disregard_frac)
            wait_times, wait_times_ts = disregard_by_length_of_interval(wait_times, departure_times, end_time,
                                                                        disregard_frac)
            queue_lengths, queue_lengths_ts = disregard_by_length_of_interval(queue_lengths, queue_length_times,
                                                                              end_time,
                                                                              disregard_frac)
            batch_mean_ages, batch_centers_age = create_nonoverlapping_batch_means(age_of_customers, age_times_ts,
                                                                                   batch_size=batch_size)
            rbatch_mean_ages = FloatVector(batch_mean_ages)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_ages)
            r_estimated_changepoint_age_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_ages)
            batch_mean_wait_times, batch_centers_wait = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                          batch_size=batch_size)
            rbatch_mean_wait_times = FloatVector(batch_mean_wait_times)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_wait_times)
            r_estimated_changepoint_wait_times_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_wait_times)
            batch_queue_lengths, batch_centers_queue = create_nonoverlapping_batch_means(queue_lengths,
                                                                                         queue_lengths_ts,
                                                                                         batch_size=batch_size)
            rbatch_mean_queue_lengths = FloatVector(batch_queue_lengths)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_queue_lengths)
            r_estimated_changepoint_queue_index = cpm_func.Detect_r_cpm_GaussianChangePoint(rbatch_mean_queue_lengths)
            estimated_changepoint_age_idx = r_estimated_changepoint_age_index[0]
            estimated_changepoint_queue_idx = r_estimated_changepoint_queue_index[0]
            estimated_changepoint_wait_times_idx = r_estimated_changepoint_wait_times_index[0]
            is_change_point_triggered_age = estimated_changepoint_age_idx > 0
            is_change_point_triggered_queue = estimated_changepoint_queue_idx > 0
            is_change_point_triggered_wait = estimated_changepoint_wait_times_idx > 0
            if np.isinf(time_of_change):
                # no changepoint
                num_no_changes += 1
                # any detection is a false positive
                if is_change_point_triggered_age:
                    if is_change_point_triggered_queue:
                        if is_change_point_triggered_wait:
                            no_changepoint_age_pos_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_pos_queue_pos_wait_neg[delta_rho] += 1
                    else:  # no quue change point
                        if is_change_point_triggered_wait:
                            no_changepoint_age_pos_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_pos_queue_neg_wait_neg[delta_rho] += 1
                else:  # no ae change point
                    if is_change_point_triggered_queue:
                        if is_change_point_triggered_wait:
                            no_changepoint_age_neg_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_neg_queue_pos_wait_neg[delta_rho] += 1
                    else:  # no queue change point
                        if is_change_point_triggered_wait:
                            no_changepoint_age_neg_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_neg_queue_neg_wait_neg[delta_rho] += 1
            else:
                num_changes += 1
                # there was a changepoint
                # still need to check if the changepoint was captured
                # a change point is captured if the detection time occurs after the event.
                dd_age = (batch_centers_age[estimated_changepoint_age_idx - 1] - time_of_change) if (
                        estimated_changepoint_age_idx > 0) else np.nan
                dd_queue = (batch_centers_queue[estimated_changepoint_queue_idx - 1] - time_of_change) if (
                        estimated_changepoint_queue_idx > 0) else np.nan
                dd_wait = (batch_centers_wait[estimated_changepoint_wait_times_idx - 1] - time_of_change) if (
                        estimated_changepoint_wait_times_idx > 0) else np.nan
                is_age_correct = False
                if not np.isnan(dd_age):
                    if dd_age > 0:
                        is_age_correct = True
                is_queue_correct = False
                if not np.isnan(dd_queue):
                    if dd_queue > 0:
                        is_queue_correct = True
                is_wait_correct = False
                if not np.isnan(dd_wait):
                    if dd_wait > 0:
                        is_wait_correct = True
                if is_age_correct:
                    if is_queue_correct:
                        if is_wait_correct:
                            changepoint_age_pos_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_pos_queue_pos_wait_neg[delta_rho] += 1
                    else:
                        if is_wait_correct:
                            changepoint_age_pos_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_pos_queue_neg_wait_neg[delta_rho] += 1
                else:
                    if is_queue_correct:
                        if is_wait_correct:
                            changepoint_age_neg_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_neg_queue_pos_wait_neg[delta_rho] += 1
                    else:
                        if is_wait_correct:
                            changepoint_age_neg_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_neg_queue_neg_wait_neg[delta_rho] += 1
    for delta_rho in delta_rhos:
        # Compute the proportions of combination of hypothesis_outcome_given change.
        prop_correct_age_pos_queue_pos_wait_pos = changepoint_age_pos_queue_pos_wait_pos[delta_rho] / max(num_changes,
                                                                                                          0.00001)
        prop_correct_age_pos_queue_pos_wait_neg = changepoint_age_pos_queue_pos_wait_neg[delta_rho] / max(num_changes,
                                                                                                          0.00001)
        prop_correct_age_pos_queue_neg_wait_pos = changepoint_age_pos_queue_neg_wait_pos[delta_rho] / max(num_changes,
                                                                                                          0.00001)
        prop_correct_age_neg_queue_pos_wait_pos = changepoint_age_neg_queue_pos_wait_pos[delta_rho] / max(num_changes,
                                                                                                          0.00001)

        prop_correct_age_pos_queue_neg_wait_neg = changepoint_age_pos_queue_neg_wait_neg[delta_rho] / max(num_changes,
                                                                                                          0.00001)
        prop_correct_age_neg_queue_neg_wait_pos = changepoint_age_neg_queue_neg_wait_pos[delta_rho] / max(num_changes,
                                                                                                          0.00001)
        prop_correct_age_neg_queue_pos_wait_neg = changepoint_age_neg_queue_pos_wait_neg[delta_rho] / max(num_changes,
                                                                                                          0.00001)
        prop_correct_age_neg_queue_neg_wait_neg = changepoint_age_neg_queue_neg_wait_neg[delta_rho] / max(num_changes,
                                                                                                          0.00001)

        prop_incorrect_age_pos_queue_pos_wait_pos = no_changepoint_age_pos_queue_pos_wait_pos[delta_rho] / max(
            num_no_changes,
            0.00001)
        prop_incorrect_age_pos_queue_pos_wait_neg = no_changepoint_age_pos_queue_pos_wait_neg[delta_rho] / max(
            num_no_changes,
            0.00001)
        prop_incorrect_age_pos_queue_neg_wait_pos = no_changepoint_age_pos_queue_neg_wait_pos[delta_rho] / max(
            num_no_changes,
            0.00001)
        prop_incorrect_age_neg_queue_pos_wait_pos = no_changepoint_age_neg_queue_pos_wait_pos[delta_rho] / max(
            num_no_changes,
            0.00001)

        prop_incorrect_age_pos_queue_neg_wait_neg = no_changepoint_age_pos_queue_neg_wait_neg[delta_rho] / max(
            num_no_changes,
            0.00001)
        prop_incorrect_age_neg_queue_neg_wait_pos = no_changepoint_age_neg_queue_neg_wait_pos[delta_rho] / max(
            num_no_changes,
            0.00001)
        prop_incorrect_age_neg_queue_pos_wait_neg = no_changepoint_age_neg_queue_pos_wait_neg[delta_rho] / max(
            num_no_changes,
            0.00001)
        prop_incorrect_age_neg_queue_neg_wait_neg = no_changepoint_age_neg_queue_neg_wait_neg[delta_rho] / max(
            num_no_changes,
            0.00001)
        values_to_add = {
            'Batch Size': batch_size,
            'rho': rho, 'delta_rho': delta_rho,
            'A+, Q+, W+ | Change': prop_correct_age_pos_queue_pos_wait_pos,
            'A+, Q+, W- | Change': prop_correct_age_pos_queue_pos_wait_neg,
            'A+, Q-, W+ | Change': prop_correct_age_pos_queue_neg_wait_pos,
            'A-, Q+, W+ | Change': prop_correct_age_neg_queue_pos_wait_pos,
            'A-, Q+, W- | Change': prop_correct_age_neg_queue_pos_wait_neg,
            'A-, Q-, W+ | Change': prop_correct_age_neg_queue_neg_wait_pos,
            'A+, Q-, W- | Change': prop_correct_age_pos_queue_neg_wait_neg,
            'A-, Q-, W- | Change': prop_correct_age_neg_queue_neg_wait_neg,
            'A+, Q+, W+ | No Change': prop_incorrect_age_pos_queue_pos_wait_pos,
            'A+, Q+, W- | No Change': prop_incorrect_age_pos_queue_pos_wait_neg,
            'A+, Q-, W+ | No Change': prop_incorrect_age_pos_queue_neg_wait_pos,
            'A-, Q+, W+ | No Change': prop_incorrect_age_neg_queue_pos_wait_pos,
            'A-, Q+, W- | No Change': prop_incorrect_age_neg_queue_pos_wait_neg,
            'A-, Q-, W+ | No Change': prop_incorrect_age_neg_queue_neg_wait_pos,
            'A+, Q-, W- | No Change': prop_incorrect_age_pos_queue_neg_wait_neg,
            'A-, Q-, W- | No Change': prop_incorrect_age_neg_queue_neg_wait_neg,
            'Run Length': end_time,
        }
        row_to_add = pd.Series(values_to_add)
        print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
        power_delay_log.write_data(data_df)
    return data_df


def simulate_joint_change_points_conditioned_on_hypothesis_outcome(data_df: pd.DataFrame, rho: List[float],
                                                                   delta_rhos: List[float],
                                                                   arr_rate_0: float, num_runs: int, start_time: float,
                                                                   end_time: float,
                                                                   my_service_rates: List[float],
                                                                   batch_size: List[int], power_delay_log: float,
                                                                   cpm_func
                                                                   ):
    """
    This code is to use the implementation of CPM in R directly from R using rpy2
    :param data_df: dataframe that will contain the performance characteristics of the test
    :param rho: list of intensity ratios
    :param delta_rhos: list of changes in intensity ratio
    :param arr_rate_0: initial arrival rate
    :param num_runs: number of runs
    :param start_time: start time of the sim 0 by default
    :param end_time: end time of the sim
    :param my_service_rates:
    :param batch_size:
    :param cpm_func: R wrapper function to cpm
    :param power_delay_log: used to save data in between runs
    :return: dataframe of the data_df
    """
    # Look at the change given positive age, queue, wait
    changepoint_age_pos_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}

    no_changepoint_age_pos_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_pos_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_pos_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_pos_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    no_changepoint_age_neg_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}

    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    # Distribution of time till false detection
    # just keep a list of all the detection times when you have a false positive
    for run_idx in range(num_runs):
        print(f"Run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            if delta_rho == 0.0:
                time_of_change = float('inf')
            else:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]
            queue_lengths, queue_length_times, mean_age_times, recording_times, wait_times, departure_times = \
                simulate_deds_return_age_queue_wait(start_time, end_time, my_arrival_rates,
                                                    time_of_changes, my_service_rates)

            age_of_customers, age_times_ts = disregard_by_length_of_interval(mean_age_times, recording_times, end_time,
                                                                             disregard_frac)
            wait_times, wait_times_ts = disregard_by_length_of_interval(wait_times, departure_times, end_time,
                                                                        disregard_frac)
            queue_lengths, queue_lengths_ts = disregard_by_length_of_interval(queue_lengths, queue_length_times,
                                                                              end_time,
                                                                              disregard_frac)
            batch_mean_ages, batch_centers_age = create_nonoverlapping_batch_means(age_of_customers, age_times_ts,
                                                                                   batch_size=batch_size)
            rbatch_mean_ages = FloatVector(batch_mean_ages)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_ages)
            r_estimated_changepoint_age_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_ages)
            batch_mean_wait_times, batch_centers_wait = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                          batch_size=batch_size)
            rbatch_mean_wait_times = FloatVector(batch_mean_wait_times)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_wait_times)
            r_estimated_changepoint_wait_times_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_wait_times)
            batch_queue_lengths, batch_centers_queue = create_nonoverlapping_batch_means(queue_lengths,
                                                                                         queue_lengths_ts,
                                                                                         batch_size=batch_size)
            rbatch_mean_queue_lengths = FloatVector(batch_queue_lengths)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_queue_lengths)
            r_estimated_changepoint_queue_index = cpm_func.Detect_r_cpm_GaussianChangePoint(rbatch_mean_queue_lengths)
            estimated_changepoint_age_idx = r_estimated_changepoint_age_index[0]
            estimated_changepoint_queue_idx = r_estimated_changepoint_queue_index[0]
            estimated_changepoint_wait_times_idx = r_estimated_changepoint_wait_times_index[0]
            is_change_point_triggered_age = estimated_changepoint_age_idx > 0
            is_change_point_triggered_queue = estimated_changepoint_queue_idx > 0
            is_change_point_triggered_wait = estimated_changepoint_wait_times_idx > 0
            if np.isinf(time_of_change):
                # no changepoint
                # any detection is a false positive
                if is_change_point_triggered_age:
                    if is_change_point_triggered_queue:
                        if is_change_point_triggered_wait:
                            no_changepoint_age_pos_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_pos_queue_pos_wait_neg[delta_rho] += 1
                    else:  # no quue change point
                        if is_change_point_triggered_wait:
                            no_changepoint_age_pos_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_pos_queue_neg_wait_neg[delta_rho] += 1
                else:  # no ae change point
                    if is_change_point_triggered_queue:
                        if is_change_point_triggered_wait:
                            no_changepoint_age_neg_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_neg_queue_pos_wait_neg[delta_rho] += 1
                    else:  # no queue change point
                        if is_change_point_triggered_wait:
                            no_changepoint_age_neg_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            no_changepoint_age_neg_queue_neg_wait_neg[delta_rho] += 1
            else:
                # there was a changepoint
                # still need to check if the changepoint was captured
                # a change point is captured if the detection time occurs after the event.
                dd_age = (batch_centers_age[estimated_changepoint_age_idx - 1] - time_of_change) if (
                        estimated_changepoint_age_idx > 0) else np.nan
                dd_queue = (batch_centers_queue[estimated_changepoint_queue_idx - 1] - time_of_change) if (
                        estimated_changepoint_queue_idx > 0) else np.nan
                dd_wait = (batch_centers_wait[estimated_changepoint_wait_times_idx - 1] - time_of_change) if (
                        estimated_changepoint_wait_times_idx > 0) else np.nan
                is_age_correct = False
                if not np.isnan(dd_age):
                    if dd_age > 0:
                        is_age_correct = True
                is_queue_correct = False
                if not np.isnan(dd_queue):
                    if dd_queue > 0:
                        is_queue_correct = True
                is_wait_correct = False
                if not np.isnan(dd_wait):
                    if dd_wait > 0:
                        is_wait_correct = True
                if is_age_correct:
                    if is_queue_correct:
                        if is_wait_correct:
                            changepoint_age_pos_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_pos_queue_pos_wait_neg[delta_rho] += 1
                    else:
                        if is_wait_correct:
                            changepoint_age_pos_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_pos_queue_neg_wait_neg[delta_rho] += 1
                else:
                    if is_queue_correct:
                        if is_wait_correct:
                            changepoint_age_neg_queue_pos_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_neg_queue_pos_wait_neg[delta_rho] += 1
                    else:
                        if is_wait_correct:
                            changepoint_age_neg_queue_neg_wait_pos[delta_rho] += 1
                        else:
                            changepoint_age_neg_queue_neg_wait_neg[delta_rho] += 1
    for delta_rho in delta_rhos:
        # Compute the proportions of correct detection given the different combinations of detections.
        prop_correct_age_pos_queue_pos_wait_pos = changepoint_age_pos_queue_pos_wait_pos[delta_rho] / max((
                changepoint_age_pos_queue_pos_wait_pos[delta_rho]
                +
                no_changepoint_age_pos_queue_pos_wait_pos[delta_rho]
        ), 0.00001)

        prop_incorrect_age_pos_queue_pos_wait_pos = no_changepoint_age_pos_queue_pos_wait_pos[delta_rho] / max((
                changepoint_age_pos_queue_pos_wait_pos[delta_rho]
                +
                no_changepoint_age_pos_queue_pos_wait_pos[delta_rho]
        ), 0.0001)
        prop_correct_age_pos_queue_pos_wait_neg = changepoint_age_pos_queue_pos_wait_neg[delta_rho] / max((
                changepoint_age_pos_queue_pos_wait_neg[delta_rho]
                +
                no_changepoint_age_pos_queue_pos_wait_neg[delta_rho]
        ), 0.0001)

        prop_incorrect_age_pos_queue_pos_wait_neg = no_changepoint_age_pos_queue_pos_wait_neg[delta_rho] / max((
                changepoint_age_pos_queue_pos_wait_neg[delta_rho]
                +
                no_changepoint_age_pos_queue_pos_wait_neg[delta_rho]
        ), 0.0001)
        prop_correct_age_pos_queue_neg_wait_pos = changepoint_age_pos_queue_neg_wait_pos[delta_rho] / max((
                changepoint_age_pos_queue_neg_wait_pos[delta_rho]
                +
                no_changepoint_age_pos_queue_neg_wait_pos[delta_rho]
        ), 0.0001)

        prop_incorrect_age_pos_queue_neg_wait_pos = no_changepoint_age_pos_queue_neg_wait_pos[delta_rho] / max((
                changepoint_age_pos_queue_neg_wait_pos[delta_rho]
                +
                no_changepoint_age_pos_queue_neg_wait_pos[delta_rho]
        ), 0.0001)
        prop_correct_age_pos_queue_neg_wait_neg = changepoint_age_pos_queue_neg_wait_neg[delta_rho] / max((
                changepoint_age_pos_queue_neg_wait_neg[delta_rho]
                +
                no_changepoint_age_pos_queue_neg_wait_neg[delta_rho]
        ), 0.0001)

        prop_incorrect_age_pos_queue_neg_wait_neg = no_changepoint_age_pos_queue_neg_wait_neg[delta_rho] / max((
                changepoint_age_pos_queue_neg_wait_neg[delta_rho]
                +
                no_changepoint_age_pos_queue_neg_wait_neg[delta_rho]
        ), 0.0001)
        prop_correct_age_neg_queue_pos_wait_pos = changepoint_age_neg_queue_pos_wait_pos[delta_rho] / max((
                changepoint_age_neg_queue_pos_wait_pos[delta_rho]
                +
                no_changepoint_age_neg_queue_pos_wait_pos[delta_rho]
        ), 0.0001)

        prop_incorrect_age_neg_queue_pos_wait_pos = no_changepoint_age_neg_queue_pos_wait_pos[delta_rho] / max((
                changepoint_age_neg_queue_pos_wait_pos[delta_rho]
                +
                no_changepoint_age_neg_queue_pos_wait_pos[delta_rho]
        ), 0.0001)
        prop_correct_age_neg_queue_pos_wait_neg = changepoint_age_neg_queue_pos_wait_neg[delta_rho] / max((
                changepoint_age_neg_queue_pos_wait_neg[delta_rho]
                +
                no_changepoint_age_neg_queue_pos_wait_neg[delta_rho]
        ), 0.0001)

        prop_incorrect_age_neg_queue_pos_wait_neg = no_changepoint_age_neg_queue_pos_wait_neg[delta_rho] / max((
                changepoint_age_neg_queue_pos_wait_neg[delta_rho]
                +
                no_changepoint_age_neg_queue_pos_wait_neg[delta_rho]
        ), 0.0001)
        prop_correct_age_neg_queue_neg_wait_pos = changepoint_age_neg_queue_neg_wait_pos[delta_rho] / max((
                changepoint_age_neg_queue_neg_wait_pos[delta_rho]
                +
                no_changepoint_age_neg_queue_neg_wait_pos[delta_rho]
        ), 0.0001)

        prop_incorrect_age_neg_queue_neg_wait_pos = no_changepoint_age_neg_queue_neg_wait_pos[delta_rho] / max((
                changepoint_age_neg_queue_neg_wait_pos[delta_rho]
                +
                no_changepoint_age_neg_queue_neg_wait_pos[delta_rho]
        ), 0.0001)
        prop_correct_age_neg_queue_neg_wait_neg = changepoint_age_neg_queue_neg_wait_neg[delta_rho] / max((
                changepoint_age_neg_queue_neg_wait_neg[delta_rho]
                +
                no_changepoint_age_neg_queue_neg_wait_neg[delta_rho]
        ), 0.0001)

        prop_incorrect_age_neg_queue_neg_wait_neg = no_changepoint_age_neg_queue_neg_wait_neg[delta_rho] / max((
                changepoint_age_neg_queue_neg_wait_neg[delta_rho]
                +
                no_changepoint_age_neg_queue_neg_wait_neg[delta_rho]
        ), 0.0001)
        values_to_add = {
            'Batch Size': batch_size,
            'rho': rho, 'delta_rho': delta_rho,
            'Change | A+, Q+, W+': prop_correct_age_pos_queue_pos_wait_pos,
            'Change | A+, Q+, W-': prop_correct_age_pos_queue_pos_wait_neg,
            'Change | A+, Q-, W+': prop_correct_age_pos_queue_neg_wait_pos,
            'Change | A-, Q+, W+': prop_correct_age_neg_queue_pos_wait_pos,
            'Change | A-, Q+, W-': prop_correct_age_neg_queue_pos_wait_neg,
            'Change | A-, Q-, W+': prop_correct_age_neg_queue_neg_wait_pos,
            'Change | A+, Q-, W-': prop_correct_age_pos_queue_neg_wait_neg,
            'Change | A-, Q-, W-': prop_correct_age_neg_queue_neg_wait_neg,
            'No Change | A+, Q+, W+': prop_incorrect_age_pos_queue_pos_wait_pos,
            'No Change | A+, Q+, W-': prop_incorrect_age_pos_queue_pos_wait_neg,
            'No Change | A+, Q-, W+': prop_incorrect_age_pos_queue_neg_wait_pos,
            'No Change | A-, Q+, W+': prop_incorrect_age_neg_queue_pos_wait_pos,
            'No Change | A-, Q+, W-': prop_incorrect_age_neg_queue_pos_wait_neg,
            'No Change | A-, Q-, W+': prop_incorrect_age_neg_queue_neg_wait_pos,
            'No Change | A+, Q-, W-': prop_incorrect_age_pos_queue_neg_wait_neg,
            'No Change | A-, Q-, W-': prop_incorrect_age_neg_queue_neg_wait_neg,
            'Run Length': end_time,
        }
        row_to_add = pd.Series(values_to_add)
        print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
        power_delay_log.write_data(data_df)
    return data_df


def simulate_change_points_in_age_process(data_df: pd.DataFrame, rho: List[float], delta_rhos: List[float],
                                          arr_rate_0: float, num_runs: int, start_time: float, end_time: float,
                                          my_service_rates: List[float],
                                          batch_size: List[int], power_delay_log: float, cpm_func,
                                          age_type="median"):
    """
    This code is to use the implementation of CPM in R directly from R using rpy2
    :param data_df: dataframe that will contain the performance characteristics of the test
    :param rho: list of intensity ratios
    :param delta_rhos: list of changes in intensity ratio
    :param arr_rate_0: initial arrival rate
    :param num_runs: number of runs
    :param start_time: start time of the sim 0 by default
    :param end_time: end time of the sim
    :param my_service_rates:
    :param my_thresholds:
    :param batch_size:
    :param cpm_func: R wrapper function to cpm
    :param power_delay_log: used to save data in between runs
    :param age_type: whether we use the mean or the median of the processes in the queue
    :return: dataframe of the data_df
    """
    true_positive = {delta_rho: 0 for delta_rho in delta_rhos}
    num_detection_dict = {delta_rho: 0 for delta_rho in delta_rhos}
    false_positive = {delta_rho: 0 for delta_rho in delta_rhos}
    false_negative = {delta_rho: 0 for delta_rho in delta_rhos}
    true_negative = {delta_rho: 0 for delta_rho in delta_rhos}
    detection_delay_dict = {delta_rho: [] for delta_rho in delta_rhos}
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    # Distribution of time till false detection
    # just keep a list of all the detection times when you have a false positive
    false_positive_list = []
    for run_idx in range(num_runs):
        print(f"Run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            if delta_rho == 0.0:
                time_of_change = float('inf')
            else:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]

            orig_ages, orig_age_time_ts = simulate_deds_return_age(start_time, end_time, my_arrival_rates,
                                                                   time_of_changes, my_service_rates, type=age_type)
            # create an artificial warm-up period
            age_of_customers, age_times_ts = disregard_by_length_of_interval(orig_ages, orig_age_time_ts, end_time,
                                                                             disregard_frac)
            if batch_size:
                batch_mean_ages, batch_centers = create_nonoverlapping_batch_means(age_of_customers, age_times_ts,
                                                                                   batch_size=batch_size)
                rbatch_mean_ages = FloatVector(batch_mean_ages)
                r.assign('remote_batch_mean_wait_times', rbatch_mean_ages)
                r_estimated_changepoint_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_ages)
            else:
                r_estimated_changepoint_index = cpm_func.Detect_r_cpm_NonParametricChangePoint(age_times_ts)

            estimated_changepoint_idx = r_estimated_changepoint_index[0]
            # print(f"CPM returned {r_estimated_changepoint_index} which should be {estimated_changepoint_idx}")

            if np.isinf(time_of_change):
                if estimated_changepoint_idx > 0:
                    num_detection_dict[delta_rho] += 1
                    false_positive[delta_rho] += 1
                    estimated_changepoint_idx -= 1  # because the index starts at one
                    if batch_size:
                        try:
                            estimated_changepoint = batch_centers[estimated_changepoint_idx]
                        except IndexError:
                            print(
                                f"Estimated changepoint idx {estimated_changepoint_idx} vs. array of length {len(batch_centers)}")
                            raise
                    else:
                        estimated_changepoint = age_times_ts[estimated_changepoint_idx]
                    false_positive_list.append(estimated_changepoint)
                else:
                    true_negative[delta_rho] += 1
            else:
                if estimated_changepoint_idx > 0:
                    estimated_changepoint_idx -= 1
                    num_detection_dict[delta_rho] += 1
                    if batch_size:
                        try:
                            estimated_changepoint = batch_centers[estimated_changepoint_idx]
                        except IndexError:
                            print(
                                f"Estimated changepoint idx {estimated_changepoint_idx} vs. array of length {len(batch_centers)}")
                    else:
                        estimated_changepoint = age_times_ts[estimated_changepoint_idx]
                    print(f"Estimated change point location: {estimated_changepoint} vs actual {time_of_change} ")
                    detection_delay = estimated_changepoint - time_of_change
                    if detection_delay >= 0:
                        true_positive[delta_rho] += 1
                        detection_delay_dict[delta_rho].append(detection_delay)
                    else:
                        false_positive[delta_rho] += 1
                        false_positive_list.append(estimated_changepoint)

                else:
                    false_negative[delta_rho] += 1
    for delta_rho in delta_rhos:
        denominator_prec = (true_positive[delta_rho] + false_positive[delta_rho])
        denominator_tp = (false_negative[delta_rho] + true_positive[delta_rho])
        denominator_fp = (false_positive[delta_rho] + true_negative[delta_rho])
        if denominator_fp:
            fp_rate = false_positive[delta_rho] / float(denominator_fp)
        else:
            fp_rate = np.nan
        if denominator_tp > 0:
            missed_detection_prob = false_negative[delta_rho] / float(denominator_tp)
            tp_rate = true_positive[delta_rho] / float(denominator_tp)
        else:
            tp_rate = np.nan
            missed_detection_prob = np.nan
        sensitivity = tp_rate
        precision = true_positive[delta_rho] / float(denominator_prec) if denominator_prec > 0 else np.nan
        prob_detect = (true_positive[delta_rho] + false_positive[delta_rho]) / float(num_runs)
        if detection_delay_dict[delta_rho]:
            mean_detection_delay = np.mean(detection_delay_dict[delta_rho])
        else:
            mean_detection_delay = np.nan
        if batch_size:
            values_to_add = {
                'Batch Size': batch_size,
                'rho': rho, 'delta_rho': delta_rho,
                'ARL_1': mean_detection_delay,
                'Missed Detection Prob': missed_detection_prob,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                'Correct_Detection': prob_detect,
                'Run Length': end_time,
                "TP": true_positive[delta_rho],
                "FP": false_positive[delta_rho],
                "FN": false_negative[delta_rho],
                "TN": true_negative[delta_rho],
                "Number Detections": num_detection_dict[delta_rho],
                "Recall": sensitivity,
                "Precision": precision,
            }
        else:
            values_to_add = {
                'Batch Size': batch_size,
                'rho': rho, 'delta_rho': delta_rho,
                'ARL_1': mean_detection_delay,
                'Missed Detection Prob': missed_detection_prob,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                'Correct_Detection': prob_detect,
                'Run Length': end_time,
                "TP": true_positive[delta_rho],
                "FP": false_positive[delta_rho],
                "FN": false_negative[delta_rho],
                "TN": true_negative[delta_rho],
                "Number Detections": num_detection_dict[delta_rho],
                "Recall": sensitivity,
                "Precision": precision,
            }

        row_to_add = pd.Series(values_to_add)
        print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
        power_delay_log.write_data(data_df)
    return data_df, false_positive_list


def simulate_changepoints_in_queuelengths_using_r_cpm(data_df: pd.DataFrame, rho: List[float], delta_rhos: List[float],
                                                      arr_rate_0: float, num_runs: int, start_time: float,
                                                      end_time: float,
                                                      my_service_rates: List[float],
                                                      batch_size: List[int], power_delay_log: float, cpm_func):
    """
    this code is to use the implementation of cpm in r directly from r using rpy2
    :param data_df: dataframe that will contain the performance characteristics of the test
    :param rho: list of intensity ratios
    :param delta_rhos: list of changes in intensity ratio
    :param arr_rate_0: initial arrival rate
    :param num_runs: number of runs
    :param start_time: start time of the sim 0 by default
    :param end_time: end time of the sim
    :param my_service_rates:
    :param my_thresholds:
    :param batch_size:
    :param cpm_func: r wrapper function to cpm
    :param power_delay_log: used to save data in between runs
    :return: dataframe of the data_df
    """
    true_positive = {delta_rho: 0 for delta_rho in delta_rhos}
    num_detection_dict = {delta_rho: 0 for delta_rho in delta_rhos}
    false_positive = {delta_rho: 0 for delta_rho in delta_rhos}
    false_negative = {delta_rho: 0 for delta_rho in delta_rhos}
    true_negative = {delta_rho: 0 for delta_rho in delta_rhos}
    detection_delay_dict = {delta_rho: [] for delta_rho in delta_rhos}
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    false_positive_list = []
    for run_idx in range(num_runs):
        print(f"run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            if delta_rho == 0.0:
                time_of_change = float('inf')
            else:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]
            orig_queue_lengths, orig_queue_lengths_ts = simulate_deds_return_queue_length(start_time, end_time,
                                                                                          my_arrival_rates,
                                                                                          time_of_changes,
                                                                                          my_service_rates)
            # create an artificial warm-up period
            queue_lengths, queue_lengths_ts = disregard_by_length_of_interval(orig_queue_lengths, orig_queue_lengths_ts,
                                                                              end_time,
                                                                              disregard_frac)
            # print(f"batch_mean_wait_times is of type {type(batch_mean_wait_times)} and length {len(batch_mean_wait_times)}")
            if batch_size:
                batch_queue_lengths, batch_centers = create_nonoverlapping_batch_means(queue_lengths,
                                                                                       queue_lengths_ts,
                                                                                       batch_size=batch_size)
                rbatch_mean_queue_lengths = FloatVector(batch_queue_lengths)
                r.assign('remote_batch_mean_wait_times', rbatch_mean_queue_lengths)
                r_estimated_changepoint_index = cpm_func.Detect_r_cpm_GaussianChangePoint(rbatch_mean_queue_lengths)
            else:
                r_estimated_changepoint_index = cpm_func.Detect_r_cpm_nonparametricchangepoint(queue_lengths)

            estimated_changepoint_idx = r_estimated_changepoint_index[0]
            # print(f"cpm returned {r_estimated_changepoint_index} which should be {estimated_changepoint_idx}")

            if np.isinf(time_of_change):
                if estimated_changepoint_idx > 0:
                    num_detection_dict[delta_rho] += 1
                    false_positive[delta_rho] += 1
                    estimated_changepoint_idx -= 1  # because the index starts at one
                    if batch_size:
                        try:
                            estimated_changepoint = batch_centers[estimated_changepoint_idx]
                        except IndexError:
                            print(
                                f"estimated changepoint idx {estimated_changepoint_idx} vs. array of length {len(batch_centers)}")
                            raise
                    else:
                        estimated_changepoint = queue_lengths_ts[estimated_changepoint_idx]
                    false_positive_list.append(estimated_changepoint)
                else:
                    true_negative[delta_rho] += 1
            else:
                if estimated_changepoint_idx > 0:
                    estimated_changepoint_idx -= 1
                    num_detection_dict[delta_rho] += 1
                    if batch_size:
                        try:
                            estimated_changepoint = batch_centers[estimated_changepoint_idx]
                        except IndexError:
                            print(
                                f"estimated changepoint idx {estimated_changepoint_idx} vs. array of length {len(batch_centers)}")
                    else:
                        estimated_changepoint = queue_lengths_ts[estimated_changepoint_idx]
                    print(f"estimated change point location: {estimated_changepoint} vs actual {time_of_change} ")
                    detection_delay = estimated_changepoint - time_of_change
                    if detection_delay >= 0:
                        true_positive[delta_rho] += 1
                        detection_delay_dict[delta_rho].append(detection_delay)
                    else:
                        false_positive[delta_rho] += 1
                        false_positive_list.append(estimated_changepoint)

                else:
                    false_negative[delta_rho] += 1
    for delta_rho in delta_rhos:
        denominator_prec = (true_positive[delta_rho] + false_positive[delta_rho])
        denominator_tp = (false_negative[delta_rho] + true_positive[delta_rho])
        denominator_fp = (false_positive[delta_rho] + true_negative[delta_rho])
        if denominator_fp:
            fp_rate = false_positive[delta_rho] / float(denominator_fp)
        else:
            fp_rate = np.nan
        if denominator_tp > 0:
            missed_detection_prob = false_negative[delta_rho] / float(denominator_tp)
            tp_rate = true_positive[delta_rho] / float(denominator_tp)
        else:
            tp_rate = np.nan
            missed_detection_prob = np.nan
        sensitivity = tp_rate
        precision = true_positive[delta_rho] / float(denominator_prec) if denominator_prec > 0 else np.nan
        prob_detect = (true_positive[delta_rho] + false_positive[delta_rho]) / float(num_runs)
        if detection_delay_dict[delta_rho]:
            mean_detection_delay = np.mean(detection_delay_dict[delta_rho])
        else:
            mean_detection_delay = np.nan
        if batch_size:
            values_to_add = {
                'batch size': batch_size,
                'rho': rho, 'delta_rho': delta_rho,
                'arl_1': mean_detection_delay,
                'missed detection prob': missed_detection_prob,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                'correct_detection': prob_detect,
                'run length': end_time,
                "tp": true_positive[delta_rho],
                "fp": false_positive[delta_rho],
                "fn": false_negative[delta_rho],
                "tn": true_negative[delta_rho],
                "number detections": num_detection_dict[delta_rho],
                "recall": sensitivity,
                "precision": precision,
            }
        else:
            values_to_add = {
                'batch size': batch_size,
                'rho': rho, 'delta_rho': delta_rho,
                'arl_1': mean_detection_delay,
                'missed detection prob': missed_detection_prob,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                'correct_detection': prob_detect,
                'run length': end_time,
                "tp": true_positive[delta_rho],
                "fp": false_positive[delta_rho],
                "fn": false_negative[delta_rho],
                "tn": true_negative[delta_rho],
                "number detections": num_detection_dict[delta_rho],
                "recall": sensitivity,
                "precision": precision,
            }

        row_to_add = pd.Series(values_to_add)
        print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
        power_delay_log.write_data(data_df)
    return data_df, false_positive_list


def plot_distribution_of_false_positive_detection_times(detection_times, batch_size, is_parametric=True):
    plt.figure()
    plt.hist(detection_times)
    plt.xlabel("Detection Times for False Alarms")
    param_string = "parametric" if is_parametric else "nonparametric"
    plt.title(f"{param_string} Change Detection for FA batch={batch_size}")
    plt.savefig(f"detection_time_distribution_batch_{batch_size}_{param_string}")
    plt.close()


def simulate_detection_delay_by_rho_delta_rho_specific_batch_size(data_df, rho, delta_rhos, arr_rate_0, num_runs,
                                                                  start_time, end_time, my_service_rates,
                                                                  my_thresholds, batch_size, power_delay_log):
    """
    vary the threshold, (pick one batch size) look at correct detection vs. false alarm see if a good ROC.
    Keep the same procedure with randomized changepoint.
    Pick one end_point
    Multiple ARL_0
    """
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    true_positive = {threshold: {delta_rho: 0 for delta_rho in delta_rhos} for threshold in my_thresholds}
    num_detection_dict = {threshold: {delta_rho: 0 for delta_rho in delta_rhos} for threshold in my_thresholds}
    false_positive = {threshold: {delta_rho: 0 for delta_rho in delta_rhos} for threshold in my_thresholds}
    false_negative_dict = {threshold: {delta_rho: 0 for delta_rho in delta_rhos} for threshold in my_thresholds}
    true_negative_dict = {threshold: {delta_rho: 0 for delta_rho in delta_rhos} for threshold in my_thresholds}
    detection_delay_dict = {threshold: {delta_rho: [] for delta_rho in delta_rhos} for threshold in my_thresholds}
    for run_idx in range(num_runs):
        print(f"Run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]
            #             wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
            #                                                                       time_of_changes, my_service_rates)
            wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates,
                                                                        time_of_changes, my_service_rates)
            batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                     batch_size=batch_size)
            for threshold in my_thresholds:
                # in GLRTChangePointDetector gamma=1/arl_0 is used to compute Ross threshold and has no effect on the
                # change point detection if use_Ross_threshold is set to False.
                arl_0 = end_time
                glrt_change_point_estimator = GLRTChangePointDetector(threshold, 1.0 / arl_0, dist_type='gaussian',
                                                                      use_Ross_threshold=False)
                glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                deds_glrt_detection_statistics = DetectionStatistics(
                    glrt_change_point_estimator.detection_result, batch_centers)
                # the default of change point time is inf
                # best guess as to when the change occurred
                # Time at which the change was detected
                detection_time = deds_glrt_detection_statistics.detection_time
                if math.isinf(time_of_change):
                    # There is no change point
                    # We have a false positive if we detected something
                    # We have a true negative if we didn't detect anything
                    if math.isinf(detection_time):
                        true_negative_dict[threshold][delta_rho] += 1
                    else:
                        false_positive[threshold][delta_rho] += 1
                else:
                    # There is a change point
                    # if the detection time is infinite then although there was a change point it went undetected
                    #       ==> false_negative
                    # if the detection occurs before the actual time of change, we have made a false detection
                    #       ==> false_positive
                    # if the detection occurs after the actual time of change, we have made a correct detection
                    #       ==> true_positive
                    detection_delay = detection_time - time_of_change
                    if math.isinf(detection_time):
                        false_negative_dict[threshold][delta_rho] += 1
                    elif detection_delay >= 0:
                        true_positive[threshold][delta_rho] += 1
                        detection_delay_dict[threshold][delta_rho].append(detection_delay)
                        num_detection_dict[threshold][delta_rho] += 1
                    else:
                        false_positive[threshold][delta_rho] += 1
                        num_detection_dict[threshold][delta_rho] += 1
    print("**** Saving ****")
    for threshold, delta_rho in itertools.product(my_thresholds, delta_rhos):
        denominator_prec = (true_positive[threshold][delta_rho] + false_positive[threshold][delta_rho])
        denominator_tp = (false_negative_dict[threshold][delta_rho] + true_positive[threshold][delta_rho])
        denominator_fp = (false_positive[threshold][delta_rho] + true_negative_dict[threshold][delta_rho])
        if denominator_fp:
            fp_rate = false_positive[threshold][delta_rho] / float(denominator_fp)
        else:
            fp_rate = np.nan
        if denominator_tp > 0:
            missed_detection_prob = false_negative_dict[threshold][delta_rho] / float(denominator_tp)
            tp_rate = true_positive[threshold][delta_rho] / float(denominator_tp)
        else:
            tp_rate = np.nan
            missed_detection_prob = np.nan
        sensitivity = tp_rate
        precision = true_positive[threshold][delta_rho] / float(denominator_prec) if denominator_prec > 0 else np.nan
        prob_detect = true_positive[threshold][delta_rho] / float(num_runs)
        if detection_delay_dict[threshold][delta_rho]:
            mean_detection_delay = np.mean(detection_delay_dict[threshold][delta_rho])
        else:
            mean_detection_delay = np.nan
        values_to_add = {
            'Batch Size': batch_size,
            'rho': rho, 'delta_rho': delta_rho,
            'ARL_1': mean_detection_delay,
            'Missed Detection Prob': missed_detection_prob,
            'h_t': threshold,
            "tp_rate": tp_rate,
            "fp_rate": fp_rate,
            'Correct_Detection': prob_detect,
            'Run Length': end_time,
            "TP": true_positive[threshold][delta_rho],
            "FP": false_positive[threshold][delta_rho],
            "FN": false_negative_dict[threshold][delta_rho],
            "TN": true_negative_dict[threshold][delta_rho],
            "Number Detections": num_detection_dict[threshold][delta_rho],
            "Recall": sensitivity,
            "Precision": precision,
        }
        row_to_add = pd.Series(values_to_add)
        print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
        power_delay_log.write_data(data_df)
    return data_df


def simulate_detection_delay_by_rho_batch_on_auto_correlation(data_df, auto_correlations, rho, delta_rho, arr_rate_0,
                                                              arl_0_list, num_runs, start_time, end_time,
                                                              my_service_rates, h_selector, sim_setting):
    """
    Based off simulate_detection_delay_by_rho_delta_rho_fixed_run_length
        Function that leverages Kemal's results on autocorrelation to pick the smallest batch size that provides
        the desired autocorrelation
    if the detection occurs after the end-time considers it as a false detection
    """
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    average_run_lengths = arl_0_list
    print(" Working on (rho={}, delta_rho={})".format(rho, delta_rho))
    arr_rate_1 = arr_rate_0 * (1 + delta_rho)
    my_arrival_rates = [arr_rate_0, arr_rate_1]
    # make the recorded data dictionaries of dictionaries
    # Because I want all of them to also depend on the batch sizes
    detection_delay_dict = {arl_0: defaultdict(list) for arl_0 in average_run_lengths}
    autocorrelation_vec_dict = {arl_0: defaultdict(list) for arl_0 in average_run_lengths}
    # autocorrelation
    checked_ac_dict = {arl_0: defaultdict(float) for arl_0 in average_run_lengths}
    # Use the following hypothesis quadrant to compute TP Rate (recall) and FP Rate  fo ROC curve
    true_positive = {arl_0: defaultdict(int) for arl_0 in average_run_lengths}
    num_detection_dict = {arl_0: defaultdict(int) for arl_0 in average_run_lengths}
    false_positive = {arl_0: defaultdict(int) for arl_0 in average_run_lengths}
    empirical_mean_threshold = {arl_0: defaultdict(list) for arl_0 in average_run_lengths}
    ross_mean_threshold = {arl_0: defaultdict(list) for arl_0 in average_run_lengths}
    false_negative_dict = {arl_0: defaultdict(int) for arl_0 in average_run_lengths}
    true_negative_dict = {arl_0: defaultdict(int) for arl_0 in average_run_lengths}
    detection_errors_dict = {arl_0: defaultdict(list) for arl_0 in average_run_lengths}
    detection_score_dict = {arl_0: defaultdict(list) for arl_0 in average_run_lengths}
    # Do multiple runs each with different ARL_0, fixed rho and multiple correlations
    arl0_batch_size_list = []
    time_of_change_dic = {arl_0: [] for arl_0 in average_run_lengths}
    number_true_changes = {arl_0: 0 for arl_0 in average_run_lengths}
    # save the number of changes and the associated score
    is_true_changes_vec = []
    likelihood_vec = []
    for arl_0 in average_run_lengths:
        for autocorrelation in auto_correlations:
            batch_selector = BatchSelectorOnAutocorrelation(autocorrelation)
            batch_size = batch_selector.return_batch_size(rho)
            # correct_batch_sizes = [1, 5, 10, 25, 50, 75, 80, 100, 125, 150, 175, 200]
            # batch_size = find_nearest(correct_batch_sizes, true_batch_size)
            detection_delay_dict[arl_0][batch_size] = []
            detection_score_dict[arl_0][batch_size] = []
            autocorrelation_vec_dict[arl_0][batch_size] = []
            false_negative_dict[arl_0][batch_size] = 0
            true_negative_dict[arl_0][batch_size] = 0
            true_positive[arl_0][batch_size] = 0
            false_positive[arl_0][batch_size] = 0
            num_detection_dict[arl_0][batch_size] = 0
            checked_ac_dict[arl_0][batch_size] = autocorrelation
            arl0_batch_size_list.append((arl_0, batch_size))
    for run_idx in range(num_runs):
        # Making different detections on the same data
        print("Run ", run_idx, "out of ", num_runs)
        for arl_0 in average_run_lengths:
            print("\t  arl_0: ", arl_0)
            if sim_setting.runtime_type == RuntimeType.RELATIVE:
                end_time = arl_0 * sim_setting.runtime_num
            elif sim_setting.runtime_type == RuntimeType.SPECIFIED:
                end_time = sim_setting.runtime_num

            if sim_setting.change_point_location_type == ChangePointLocationChoice.SPECIFIED:
                time_of_change = sim_setting.change_point_location
            elif sim_setting.change_point_location_type == ChangePointLocationChoice.UNIFORM:
                time_of_change = random.randrange(5, int(end_time))
            elif sim_setting.change_point_location_type == ChangePointLocationChoice.RELATIVE:
                time_of_change = sim_setting.change_point_location
            elif sim_setting.change_point_location_type == ChangePointLocationChoice.GEOMETRIC:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            else:
                time_of_change = int(0.5 * end_time)
            time_of_change_dic[arl_0].append(time_of_change)
            # ensure the end time is after the true change point
            if not math.isinf(time_of_change):
                number_true_changes[arl_0] += 1
            time_of_changes = [-1, time_of_change]
            print("\t\t Time of change: ", time_of_change)
            #             wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates,
            #                                                                         time_of_changes, my_service_rates)
            wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
                                                                      time_of_changes, my_service_rates)
            for autocorrelation in auto_correlations:
                print("\t\t\t Autocorrelation: {} and rho: {}".format(autocorrelation, rho))
                batch_selector = BatchSelectorOnAutocorrelation(autocorrelation)
                true_batch_size = batch_selector.return_batch_size(rho)
                # correct_batch_sizes = [1, 5, 10, 25, 50, 75, 80, 100, 125, 150, 175, 200]
                # batch_size = find_nearest(correct_batch_sizes, true_batch_size)
                batch_size = true_batch_size
                print("\t\t\t Batch size:{} has been approximated to {}".format(true_batch_size, batch_size))
                batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                         batch_size=batch_size)
                threshold = h_selector.get_threshold(batch_size, arl_0)
                empirical_mean_threshold[arl_0][batch_size].append(threshold)
                # 3. Run a change point detection algorithm on the m
                # Use Ross' equation 5
                glrt_change_point_estimator = GLRTChangePointDetector(threshold, 1.0 / arl_0, dist_type='gaussian',
                                                                      use_Ross_threshold=False)
                ross_threshold = glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
                ross_mean_threshold[arl_0][batch_size].append(ross_threshold)
                deds_glrt_detection_statistics = DetectionStatistics(
                    glrt_change_point_estimator.detection_result, batch_centers)
                # the default of change point time is inf
                # best guess as to when the change occurred
                glrt_detected_change_time = deds_glrt_detection_statistics.change_point_time
                # Time at which the change was detected
                detection_time = deds_glrt_detection_statistics.detection_time
                # There is a change if the change time is greater than time of changes
                if math.isinf(time_of_change):
                    # There is no change point
                    # We have a false positive if we detected something
                    # We have a true negative if we didn't detect anything
                    if math.isinf(detection_time):
                        true_negative_dict[threshold][delta_rho] += 1
                    else:
                        false_positive[threshold][delta_rho] += 1
                else:
                    # There is a change point
                    # if the detection time is infinite then although there was a change point it went undetected
                    #       ==> false_negative
                    # if the detection occurs before the actual time of change, we have made a false detection
                    #       ==> false_positive
                    # if the detection occurs after the actual time of change, we have made a correct detection
                    #       ==> true_positive
                    detection_delay = detection_time - time_of_change
                    if math.isinf(detection_time):
                        false_negative_dict[threshold][delta_rho] += 1
                    elif detection_delay >= 0:
                        true_positive[threshold][delta_rho] += 1
                        detection_delay_dict[threshold][delta_rho].append(detection_delay)
                        num_detection_dict[threshold][delta_rho] += 1
                    else:
                        false_positive[threshold][delta_rho] += 1
                        num_detection_dict[threshold][delta_rho] += 1
                detection_error = glrt_detected_change_time - time_of_change
                detection_errors_dict[arl_0][batch_size].append(detection_error)
                is_true_changes_vec.append(not math.isinf(time_of_change))
                likelihood_vec.append(glrt_change_point_estimator.detection_result.detection_value)
                detection_score_dict[arl_0][batch_size].append(
                    glrt_change_point_estimator.detection_result.detection_value)
                autocorrelation_vec_dict[arl_0][batch_size].append(deds_glrt_detection_statistics.autocorrelation)
            # end autocorrelation
        # end arl_0 for loop

    # end run_idx for loop
    print("**** Saving ****")
    for arl_0, batch_val in arl0_batch_size_list:
        denominator_1 = (true_positive[arl_0][batch_val] + false_positive[arl_0][batch_val])
        denominator_2 = (false_negative_dict[arl_0][batch_val] + true_positive[arl_0][batch_val])
        denominator_3 = (false_positive[arl_0][batch_val] + true_negative_dict[arl_0][batch_val])
        if denominator_3:
            fa_rate = false_positive[arl_0][batch_val] / float(denominator_3)
        else:
            fa_rate = np.nan
        if denominator_2 > 0:
            missed_detection_prob = false_negative_dict[arl_0][batch_val] / float(denominator_2)
            tp_rate = true_positive[arl_0][batch_val] / float(denominator_2)
        else:
            tp_rate = np.nan
            missed_detection_prob = np.nan
        sensitivity = tp_rate
        precision = true_positive[arl_0][batch_val] / float(denominator_1) if denominator_1 > 0 else np.nan
        prob_detect = true_positive[arl_0][batch_val] / float(num_runs)
        if len(autocorrelation_vec_dict[arl_0][batch_val]):
            mean_acf1 = np.mean(autocorrelation_vec_dict[arl_0][batch_val])
            std_acf1 = np.std(autocorrelation_vec_dict[arl_0][batch_val])
        else:
            mean_acf1 = np.nan
            std_acf1 = np.nan
        std_detection_delay = np.nan
        median_detection_delay = np.nan
        mean_detection_delay = np.nan
        if len(detection_delay_dict[arl_0][batch_val]):
            try:
                std_detection_delay = np.std(detection_delay_dict[arl_0][batch_val])
                median_detection_delay = np.median(detection_delay_dict[arl_0][batch_val])
                # This line fails if during all the tests no detection was made
                mean_detection_delay = np.mean(detection_delay_dict[arl_0][batch_val])
            except RuntimeWarning:
                print("Issue with sim for arl_0={} and batch_val={}".format(arl_0, batch_val))
        else:
            median_detection_delay = float('inf')
            std_detection_delay = float('inf')
            mean_detection_delay = float("inf")
        # prepare what you want to add to the dataframe
        #         print(detection_errors_dict[arl_0][batch_val])
        values_to_add = {'ARL_0': arl_0, 'Batch Size': batch_val, 'rho': rho, 'delta_rho': delta_rho,
                         'ARL_1': mean_detection_delay,
                         "Avg Detection Error": np.nanmean(detection_errors_dict[arl_0][batch_val]),
                         'DDelay_std': std_detection_delay,
                         'DDelay_median': median_detection_delay,
                         'Missed Detection Prob': missed_detection_prob,
                         "tp_rate": tp_rate, "fa_rate": fa_rate,
                         'Set Autocorrelation': checked_ac_dict[arl_0][batch_val],
                         'Mean Lag 1 AutoCorrelation': mean_acf1, "Std Lag 1 AutoCorrelation": std_acf1,
                         'Correct_Detection': prob_detect, 'Run Length': end_time,
                         'Time of change': np.median(time_of_change_dic[arl_0]),
                         'NumberTrueChanges': number_true_changes[arl_0],
                         "TP": true_positive[arl_0][batch_val], "FP": false_positive[arl_0][batch_val],
                         "FN": false_negative_dict[arl_0][batch_val],
                         "TN": true_negative_dict[arl_0][batch_val],
                         "Number Detections": num_detection_dict[arl_0][batch_val],
                         "Ross_ht": np.mean(ross_mean_threshold[arl_0][batch_val]),
                         "Empirical_ht": np.mean(empirical_mean_threshold[arl_0][batch_val]),
                         "Recall": sensitivity,
                         "Precision": precision,
                         "G_n": np.median(detection_score_dict[arl_0][batch_val])
                         }

        row_to_add = pd.Series(values_to_add)
        # print(row_to_add)
        data_df = data_df.append(row_to_add, ignore_index=True)
    # plot_density_histogram_of_detections(is_true_changes_vec, likelihood_vec, title=f"rho={rho}, deltarho={delta_rho}")
    return data_df


# def simulate_detection_delay_by_rho_delta_rho_fixed_run_length(data_df, rho, delta_rho, arr_rate_0, arl_0_list,
#                                                                batch_sizes,
#                                                                num_runs, start_time, end_time,
#                                                                my_service_rates,
#                                                                h_selector,
#                                                                change_point_location_type,
#                                                                change_point_location):
#     average_run_lengths = arl_0_list
#     print(" Working on (rho={}, delta_rho={})".format(rho, delta_rho))
#     arr_rate_1 = arr_rate_0 * (1 + delta_rho)
#     my_arrival_rates = [arr_rate_0, arr_rate_1]
#     detection_delay_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     false_detection_rates_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     missed_detection_dict = {batch: defaultdict(int) for batch in batch_sizes}
#     probability_correct_detection_no_false_alarm = {batch: defaultdict(int) for batch in batch_sizes}
#     for run_idx in range(num_runs):
#         print("Change point type ", change_point_location_type, " at: ", end="\t")
#         if change_point_location_type == ChangePointLocationChoice.UNIFORM:
#             time_of_change = random.randrange(5, int(end_time))
#         elif change_point_location_type == ChangePointLocationChoice.RELATIVE:
#             time_of_change = change_point_location * end_time
#         elif change_point_location_type == ChangePointLocationChoice.SPECIFIED:
#             time_of_change = change_point_location
#         else:
#             raise Exception("Invalid Simulation Setting Change Point Location Input")
#         time_of_changes = [-1, time_of_change]
#         print("Time of changes ", time_of_changes)
#         print("End time ", end_time)
# #         wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates,
# #                                                                     time_of_changes, my_service_rates)
#         wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
#                                                                   time_of_changes, my_service_rates)
#         for batch_size in batch_sizes:
#             batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
#                                                                                      batch_size=batch_size)
#             for arl_0 in average_run_lengths:
#                 print("\t batch={} and arl={}".format(batch_size, arl_0))
#                 threshold = h_selector.get_threshold(batch_size, arl_0)
#                 # 3. Run a change point detection algorithm on the m
#                 glrt_change_point_estimator = GLRTChangePointDetector(threshold, 1/arl_0, dist_type='gaussian')
#                 glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
#                 deds_glrt_detection_statistics = DetectionStatistics(
#                     glrt_change_point_estimator.detection_result, batch_centers)
#                 glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
#                 if len(glrt_detected_change_times):
#                     is_time_detected_before_change = glrt_detected_change_times[0] < time_of_changes[1]
#                 else:
#                     is_time_detected_before_change = False
#                 detection_index = find_detection_index(glrt_detected_change_times, time_of_changes[1])
#                 if len(glrt_detected_change_times) and 0 <= detection_index < len(glrt_detected_change_times):
#                     detection_delay = glrt_detected_change_times[detection_index] - time_of_changes[1]
#                     assert (detection_delay >= 0)
#                     false_detection_list = glrt_detected_change_times[:detection_index] \
#                                            + glrt_detected_change_times[detection_index + 1:]
#                     if is_time_detected_before_change:
#                         probability_correct_detection_no_false_alarm[batch_size][arl_0] += 1
#                     probability_correct_detection[batch_size][arl_0] += 1
#                 else:  # nothing was detected
#                     detection_delay = float("inf")  # change that to float('inf')
#                     false_detection_list = glrt_detected_change_times
#                     missed_detection_dict[batch_size][arl_0] += 1
#
#                 if len(glrt_detected_change_times) == 0:
#                     logging.warning("No change detected for a change point at {} and end-time at {}".format(
#                         time_of_change, end_time
#                     ))
#                 false_detection_prob = len(false_detection_list) / float(max(len(glrt_detected_change_times), 1))
#                 false_detection_rates_dict[batch_size][arl_0].append(false_detection_prob)
#                 detection_delay_dict[batch_size][arl_0].append(detection_delay)
#                 # end arl for-loop
#             # end batch_size for loop
#         # end run_idx for loop
#     for arl_0 in average_run_lengths:
#         for batch in batch_sizes:
#             mean_false_detection_prob = np.nanmean(false_detection_rates_dict[batch][arl_0])
#             if len(detection_delay_dict[batch][arl_0]):
#                 mean_detection_delay = np.median(detection_delay_dict[batch][arl_0])
#                 try:
#                     std_detection_delay = np.std(detection_delay_dict[batch][arl_0])
#                 except RuntimeWarning:
#                     std_detection_delay = np.nan
#             else:
#                 mean_detection_delay = float('inf')
#                 std_detection_delay = float("inf")
#             missed_detection_prob = missed_detection_dict[batch][arl_0] / float(num_runs)
#             prob_detect_b4_alarm = probability_correct_detection_no_false_alarm[batch][arl_0] / float(num_runs)
#             prob_detect = probability_correct_detection[batch][arl_0] / float(num_runs)
#             # prepare what you want to add to the dataframe
#             values_to_add = {'ARL_0': arl_0, 'Batch Size': batch, 'rho': rho, 'delta_rho': delta_rho,
#                              'ARL_1': mean_detection_delay, 'ARL_1_std': std_detection_delay,
#                              'False Detection Rate': mean_false_detection_prob,
#                              'Conditional_Correct_Detection': prob_detect_b4_alarm,
#                              'Correct_Detection': prob_detect, 'Run Length': end_time,
#                              'Time of Change': "uniform"
#                              if change_point_location_type == ChangePointLocationChoice.UNIFORM
#                              else change_point_location}
#             row_to_add = pd.Series(values_to_add)
#             # print(row_to_add)
#             print("***Saving***")
#             data_df = data_df.append(row_to_add, ignore_index=True)
#     return data_df
#
#
# def simulate_detection_delay_by_rho_delta_rho_arl_0_run_len(data_df, rho, delta_rho, arr_rate_0, arl_0_list,
#                                                             batch_sizes, num_runs, start_time,
#                                                             my_service_rates, h_selector, sim_setting):
#     average_run_lengths = arl_0_list
#     print("\t ******* ARL_0 defined run length ********")
#     print(" Working on (rho={}, delta_rho={})".format(rho, delta_rho))
#     arr_rate_1 = arr_rate_0 * (1 + delta_rho)
#     my_arrival_rates = [arr_rate_0, arr_rate_1]
#     detection_delay_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     false_detection_rates_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     missed_detection_dict = {batch: defaultdict(int) for batch in batch_sizes}
#     probability_correct_detection_no_false_alarm = {batch: defaultdict(int) for batch in batch_sizes}
#     probability_correct_detection = {batch: defaultdict(int) for batch in batch_sizes}
#     for run_idx in range(num_runs):
#         for arl_0 in average_run_lengths:
#             print("Setting the end time to: ", arl_0)
#             if sim_setting.change_point_location_type == ChangePointLocationChoice.UNIFORM:
#                 time_of_change = random.randrange(5, arl_0 - 5)
#             elif sim_setting.change_point_location_type == ChangePointLocationChoice.RELATIVE:
#                 time_of_change = sim_setting.change_point_location * arl_0
#             else:
#                 time_of_change = sim_setting.change_point_location
#             time_of_changes = [-1, time_of_change]
# #             wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, arl_0, my_arrival_rates,
# #                                                                         time_of_changes, my_service_rates)
#             wait_times, wait_times_ts = simulate_ladder_point_process(start_time, arl_0, my_arrival_rates,
#                                                                       time_of_changes, my_service_rates)
#             for batch_size in batch_sizes:
#                 batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
#                                                                                          batch_size=batch_size)
#                 threshold = h_selector.get_threshold(batch_size, arl_0)
#                 # 3. Run a change point detection algorithm on the m
#                 glrt_change_point_estimator = GLRTChangePointDetector(threshold, 1/arl_0, dist_type='gaussian')
#                 glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
#                 deds_glrt_detection_statistics = DetectionStatistics(
#                     glrt_change_point_estimator.detection_result, batch_centers)
#                 glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
#                 if len(glrt_detected_change_times):
#                     is_time_detected_before_change = glrt_detected_change_times[0] < time_of_changes[1]
#                 else:
#                     is_time_detected_before_change = False
#                 detection_index = find_detection_index(glrt_detected_change_times, time_of_changes[1])
#                 if len(glrt_detected_change_times) and 0 <= detection_index < len(glrt_detected_change_times):
#                     detection_delay = glrt_detected_change_times[detection_index] - time_of_changes[1]
#                     assert (detection_delay >= 0)
#                     false_detection_list = glrt_detected_change_times[:detection_index] \
#                                            + glrt_detected_change_times[detection_index + 1:]
#                     if is_time_detected_before_change:
#                         probability_correct_detection_no_false_alarm[batch_size][arl_0] += 1
#                     probability_correct_detection[batch_size][arl_0] += 1
#                 else:  # nothing was detected
#                     detection_delay = float("inf")  # change that to float('inf')
#                     false_detection_list = glrt_detected_change_times
#                     missed_detection_dict[batch_size][arl_0] += 1
#                 false_detection_prob = len(false_detection_list) / float(len(batch_mean_wait_times))
#                 false_detection_rates_dict[batch_size][arl_0].append(false_detection_prob)
#                 detection_delay_dict[batch_size][arl_0].append(detection_delay)
#                 # end arl for-loop
#             # end batch_size for loop
#         # end run_idx for loop
#     for arl_0 in average_run_lengths:
#         for batch in batch_sizes:
#             mean_false_detection_prob = np.nanmean(false_detection_rates_dict[batch][arl_0])
#             if len(detection_delay_dict[batch][arl_0]):
#                 mean_detection_delay = np.median(detection_delay_dict[batch][arl_0])
#                 try:
#                     std_detection_delay = np.std(detection_delay_dict[batch][arl_0])
#                 except RuntimeWarning:
#                     std_detection_delay = np.nan
#             else:
#                 mean_detection_delay = float('inf')
#                 std_detection_delay = float('inf')
#             missed_detection_prob = missed_detection_dict[batch][arl_0] / float(num_runs)
#             prob_detect_b4_alarm = probability_correct_detection_no_false_alarm[batch][arl_0] / float(num_runs)
#             prob_detect = probability_correct_detection[batch][arl_0] / float(num_runs)
#             # prepare what you want to add to the dataframe
#             values_to_add = {'ARL_0': arl_0, 'Batch Size': batch, 'rho': rho, 'delta_rho': delta_rho,
#                              'ARL_1': mean_detection_delay, 'ARL_1_std': std_detection_delay,
#                              'False Detection Rate': mean_false_detection_prob,
#                              'Conditional_Correct_Detection': prob_detect_b4_alarm,
#                              'Correct_Detection': prob_detect, 'Run Length': arl_0}
#             row_to_add = pd.Series(values_to_add)
#             # print(row_to_add)
#             data_df = data_df.append(row_to_add, ignore_index=True)
#     return data_df
#
#
# def simulate_detection_delay_with_window(data_df, rho, delta_rho, arr_rate_0, average_run_lengths, batch_sizes,
#                                          num_runs, start_time, my_service_rates, h_selector,
#                                          sim_setting: SimulationSetting,
#                                          detection_lengths):
#     arr_rate_1 = arr_rate_0 * (1 + delta_rho)
#     my_arrival_rates = [arr_rate_0, arr_rate_1]
#     detection_delay_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     # if we don't track false detections
#     detection_delay_noinf_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     false_detection_rates_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     autocorrelation_vec_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     db_vec_dict = {batch: {arl_0: [] for arl_0 in average_run_lengths} for batch in batch_sizes}
#     missed_detection_dict = {batch: defaultdict(int) for batch in batch_sizes}
#     # they are also indexed by detection lengths, the time it takes after a changepoint to detect that change
#     probability_correct_detection_no_false_alarm = {detection_time:{batch: defaultdict(int) for batch in batch_sizes}
#                                                     for detection_time in detection_lengths}
#     probability_correct_detection = {detection_time: {batch: defaultdict(int) for batch in batch_sizes}
#                                      for detection_time in detection_lengths}
#     run_lenghts = {arl_0: sim_setting.runtime_num for arl_0 in average_run_lengths}
#     for run_idx in range(num_runs):
#         for arl_0 in average_run_lengths:
#             print("Run ", run_idx,  "out of ", num_runs,  " arl_0: ", arl_0)
#             if sim_setting.runtime_type == RuntimeType.RELATIVE:
#                 end_time = arl_0 * sim_setting.runtime_num
#             elif sim_setting.runtime_type == RuntimeType.SPECIFIED:
#                 end_time = sim_setting.runtime_num
#             else:
#                 end_time = sim_setting.runtime_num
#             run_lenghts[arl_0] = end_time
#             if sim_setting.change_point_location_type == ChangePointLocationChoice.UNIFORM:
#                 time_of_change = random.randrange(5, arl_0 - 5)
#             elif sim_setting.change_point_location_type == ChangePointLocationChoice.RELATIVE:
#                 time_of_change = sim_setting.change_point_location * arl_0
#             else:
#                 time_of_change = sim_setting.change_point_location
#
#             time_of_changes = [-1, time_of_change]
#             print("Time of change: ", time_of_change)
# #             wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates,
# #                                                                         time_of_changes, my_service_rates)
#             wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
#                                                                         time_of_changes, my_service_rates)
#             for batch_size in batch_sizes:
#                 print("\t Batch size: ", batch_size)
#                 batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
#                                                                                          batch_size=batch_size)
#                 threshold = h_selector.get_threshold(batch_size, arl_0)
#                 # 3. Run a change point detection algorithm on the m
#                 glrt_change_point_estimator = GLRTChangePointDetector(threshold, 1 / arl_0, dist_type='gaussian')
#                 glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
#                 deds_glrt_detection_statistics = DetectionStatistics(
#                     glrt_change_point_estimator.detection_result, batch_centers)
#                 glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
#                 if len(glrt_detected_change_times):
#                     is_time_detected_before_change = glrt_detected_change_times[0] < time_of_changes[1]
#                 else:
#                     is_time_detected_before_change = False
#                 detection_index = find_detection_index(glrt_detected_change_times, time_of_changes[1])
#                 candidate_change_times = glrt_detected_change_times[detection_index:]
#                 if len(glrt_detected_change_times) and 0 <= detection_index < len(glrt_detected_change_times):
#                     detection_time = glrt_detected_change_times[detection_index]
#                     detection_delay = detection_time - time_of_changes[1]
#                     assert (detection_delay >= 0)
#                     false_detection_list = glrt_detected_change_times[:detection_index] \
#                                            + glrt_detected_change_times[detection_index + 1:]
#                     for det_time in detection_lengths:
#                         if (det_time+time_of_change) >= detection_time:
#                             probability_correct_detection[det_time][batch_size][arl_0] += 1
#                             if is_time_detected_before_change:
#                                 probability_correct_detection_no_false_alarm[det_time][batch_size][arl_0] += 1
#                 else:  # nothing was detected
#                     detection_delay = float("inf")  # change that to float('inf')
#                     false_detection_list = glrt_detected_change_times
#                     missed_detection_dict[batch_size][arl_0] += 1
#                 false_detection_prob = len(false_detection_list) / float(len(batch_mean_wait_times))
#                 false_detection_rates_dict[batch_size][arl_0].append(false_detection_prob)
#                 detection_delay_dict[batch_size][arl_0].append(detection_delay)
#                 if detection_delay < float("inf"):
#                     detection_delay_noinf_dict[batch_size][arl_0].append(detection_delay)
#                 if len(deds_glrt_detection_statistics.autocorrelation_vector) > 0:
#                     autocorrelation_vec_dict[batch_size][arl_0].append(
#                         np.mean(deds_glrt_detection_statistics.autocorrelation_vector))
#                     db_vec_dict[batch_size][arl_0].append(
#                         np.mean(deds_glrt_detection_statistics.db_statistics))
#
#                 # end arl for-loop
#             # end batch_size for loop
#         # end run_idx for loop
#     print("**** Saving ****")
#     for arl_0 in average_run_lengths:
#         for batch in batch_sizes:
#             mean_acf1 = np.mean(autocorrelation_vec_dict[batch][arl_0])
#             mean_db_stat = np.nanmean(db_vec_dict[batch][arl_0])
#             std_acf1 = np.std(autocorrelation_vec_dict[batch][arl_0])
#             mean_false_detection_prob = np.nanmean(false_detection_rates_dict[batch][arl_0])
#             mean_detection_delay_no_inf = np.nan
#             std_detection_delay_no_inf = np.nan
#             if len(detection_delay_dict[batch][arl_0]):
#                 try:
#                     std_detection_delay = np.std(detection_delay_dict[batch][arl_0])
#                     std_detection_delay_no_inf = np.std(detection_delay_noinf_dict[batch][arl_0])
#                 except RuntimeWarning:
#                     std_detection_delay = np.nan
#                     std_detection_delay_no_inf = np.nan
#                 median_detection_delay = np.median(detection_delay_dict[batch][arl_0])
#                 mean_detection_delay_no_inf = np.mean(detection_delay_noinf_dict[batch][arl_0])
#             else:
#                 median_detection_delay = float('inf')
#                 std_detection_delay = float('inf')
#             missed_detection_prob = missed_detection_dict[batch][arl_0] / float(num_runs)
#             for det_time in detection_lengths:
#                 prob_detect_b4_alarm = probability_correct_detection_no_false_alarm[det_time][batch][arl_0] / float(num_runs)
#                 prob_detect = probability_correct_detection[det_time][batch][arl_0] / float(num_runs)
#                 # prepare what you want to add to the dataframe
#                 values_to_add = {'ARL_0': arl_0, 'Batch Size': batch, 'rho': rho, 'delta_rho': delta_rho,
#                                  'ARL_1': median_detection_delay, 'ARL_1_std': std_detection_delay,
#                                  'DDelay_mean': mean_detection_delay_no_inf, 'DDelay_std': std_detection_delay_no_inf,
#                                  'False Detection Rate': mean_false_detection_prob,
#                                  'Mean Lag 1 AutoCorrelation': mean_acf1, "Std Lag 1 AutoCorrelation": std_acf1,
#                                  "Durbin-Watson": mean_db_stat,
#                                  'Conditional_Correct_Detection': prob_detect_b4_alarm,
#                                  'Correct_Detection': prob_detect, 'Run Length': run_lenghts[arl_0],
#                                  "Detection Time": det_time}
#                 row_to_add = pd.Series(values_to_add)
#             # print(row_to_add)
#                 data_df = data_df.append(row_to_add, ignore_index=True)
#     return data_df


# def simulate_detection_delays(batch_sizes, average_run_lengths, rho_vec, delta_rho_rel, mu_baseline, num_runs,
#                               h_selector, power_delay_log, sim_setting,
#                               detection_windows=None):
#     """
#         Simulate detection delays by picking the right threshold based on the desired average run length
#         to false detection
#         :param batch_sizes: set of batch sizes to try out
#         :param average_run_lengths: run lengths of the different tests as ARL_0
#         :param rho_vec: vector of intensity ratios
#         :param delta_rho_rel: vector of relative increments in intensity ratio
#         :param mu_baseline: baseline service rate
#         :param num_runs: number of runs to play
#         :param h_selector: Threshold functor for the detection test
#         :param power_delay_log: object used to save the statistics of the test
#         :param sim_setting: object of the class SimulationSetting
#         :param detection_windows: Not used
#         :return: dataframe
#     """
#     print("Simulating {} runs".format(num_runs))
#     mu = mu_baseline
#     start_time = 0
#     my_service_rates = [mu, mu]
#     data_store = pd.DataFrame()
#     if sim_setting.use_detection_window:
#         print("There is a detection window")
#         rho = rho_vec[0]
#         delta_rho = delta_rho_rel[0]
#         print("Running Detection Length Test")
#         print("Current rho={0:2.4f}, delta rho={1:2.4f}".format(rho, delta_rho))
#         arr_rate_0 = mu * rho
#         data_store = simulate_detection_delay_with_window(data_store, rho, delta_rho, arr_rate_0, average_run_lengths,
#                                                           batch_sizes, num_runs, start_time, my_service_rates,
#                                                           h_selector, sim_setting, detection_windows)
#         power_delay_log.write_data(data_store)
#     else:
#         print("No detection window")
#         for rho_idx, rho in enumerate(rho_vec):
#             arr_rate_0 = mu * rho
#             for delta_rho_idx, delta_rho in enumerate(delta_rho_rel):
#                 # end delta_rho
#                 print("Current rho={0:2.4f}, delta rho={1:2.4f}".format(rho, delta_rho))
#                 if sim_setting.runtime_type == RuntimeType.RELATIVE:
#                     data_store = simulate_detection_delay_by_rho_delta_rho_arl_0_run_len(data_store, rho, delta_rho,
#                                                                                          arr_rate_0,
#                                                                                          average_run_lengths,
#                                                                                          batch_sizes, num_runs,
#                                                                                          start_time,
#                                                                                          my_service_rates, h_selector,
#                                                                                          sim_setting)
#                     power_delay_log.write_data(data_store)
#                 else:
#                     print(sim_setting.runtime_type, RuntimeType.SPECIFIED)
#                     if sim_setting.runtime_type == RuntimeType.SPECIFIED:
#                         end_times = [sim_setting.runtime_num]
#                     elif sim_setting.runtime_type == RuntimeType.LIST:
#                         end_times = sim_setting.runtime_num
#                     else:
#                         raise Exception("Bad Runtime input")
#
#                     for t_end in end_times:
#                         print("Current end time: ", t_end)
#                         if sim_setting.change_point_location_type == ChangePointLocationChoice.SPECIFIED:
#                             if isinstance(sim_setting.change_point_location, list):
#                                 for change_point_loc in sim_setting.change_point_location:
#                                     data_store = simulate_detection_delay_by_rho_delta_rho_fixed_run_length(data_store,
#                                                                                                             rho,
#                                                                                                             delta_rho,
#                                                                                                             arr_rate_0,
#                                                                                                             average_run_lengths,
#                                                                                                             batch_sizes,
#                                                                                                             num_runs,
#                                                                                                             start_time,
#                                                                                                             t_end,
#                                                                                                             my_service_rates,
#                                                                                                             h_selector,
#                                                                                                             sim_setting.change_point_location_type,
#                                                                                                             change_point_loc)
#                             else:
#                                 data_store = simulate_detection_delay_by_rho_delta_rho_fixed_run_length(data_store, rho,
#                                                                                                         delta_rho, arr_rate_0,
#                                                                                                         average_run_lengths,
#                                                                                                         batch_sizes, num_runs,
#                                                                                                         start_time, t_end,
#                                                                                                         my_service_rates,
#                                                                                                         h_selector,
#                                                                                                         sim_setting.change_point_location_type,
#                                                                                                         sim_setting.change_point_location)
#                         else:
#                             data_store = simulate_detection_delay_by_rho_delta_rho_fixed_run_length(data_store, rho,
#                                                                                                     delta_rho,
#                                                                                                     arr_rate_0,
#                                                                                                     average_run_lengths,
#                                                                                                     batch_sizes,
#                                                                                                     num_runs,
#                                                                                                     start_time, t_end,
#                                                                                                     my_service_rates,
#                                                                                                     h_selector,
#                                                                                                     sim_setting.change_point_location_type,
#                                                                                                     sim_setting.change_point_location)
#
#                         power_delay_log.write_data(data_store)
#             # end rho
#     return data_store
#
#
# def main():
#     setting = SimulationSetting()
#     setting.runtime_type = RuntimeType.SPECIFIED
#     setting.runtime_num = 20000
# #     setting.change_point_location_type = ChangePointLocationChoice.UNIFORM
#     setting.change_point_location_type = ChangePointLocationChoice.SPECIFIED
#     setting.change_point_location = [5000, 10000, 15000]
#     setting.use_detection_window = False
#     selected_batch_sizes = [1, 5, 10, 25, 50, 80, 100, 200]
#     # selected_average_run_lengths = [25000, 100000,  150000]
#     selected_average_run_lengths = [25000]
# #     delta_rho_rel = np.logspace(-2, 2, 5)
#     delta_rho_rel = [0.25, 0.5, 0.75, 1]
#     # rho = np.linspace(0.5, 1.2, 4)
#     rho = [0.25, 0.5, 0.75]
#     # rho = [0.75] # redo the sim for that missing rho
#     mu_baseline = 1
#     num_runs = 600
#     pkl_directory = "./Results/GLRT_ROSS/ARL_0/"
#     h_selector = ThresholdSelector(pkl_directory)
#
#     log_directory = "./Results/GLRT_ROSS/ARL_1/"
#     log_file_name = log_directory + "select_detection_delay_test_log_"
#     data_store = pd.DataFrame(columns=['ARL_0', 'Batch Size', 'rho', 'delta_rho', 'ARL_1', 'False Detection Rate',
#                                        'Conditional_Correct_Detection', 'Correct_Detection'])
#     power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
#
#     # Run the simulation with a uniformly distributed change point over the run length
#     detection_delay_df = simulate_detection_delays(selected_batch_sizes, selected_average_run_lengths, rho,
#                                                    delta_rho_rel, mu_baseline, num_runs, h_selector, power_delay_log,
#                                                    setting)
#     # plot_facet_semilogx_conditional_correct_detection(detection_delay_df, "./Figures/ARL_1/JP_Facet_cond_correct_2.png")
#     # plot_facet_semilogx_correct_detection(detection_delay_df, "./Figures/ARL_1/JP_Facet_correct_2.png")
#     print(detection_delay_df.tail())
#     # Make two types of plot
#     # 1. a facet grid of scatterplots
# #    plot_facet_grid(detection_delay_df, "./Figures/ARL_1/Facet_Grid_delay_vs_rho_and_delta-rho.png")
#     # 2. multiple 3d plots for different batch size and arl_0 with rho delta_rho
# #    plot_three_d_plots(detection_delay_df, "./Figures/ARL_1/3D_delay_vs_rho_and_delta-rho_")
#
#
# def main_2():
#     """
#     Dave Goldsman's experiment
#     Test to see how long a test needs to run to find a test
#     Probability of correct detection for a given allowed length after a change point
#
#     """
#     # Fix rho and delta rho
#     rho = [0.5, 0.75, 0.8]
#     delta_rho_rel = [0.25, 0.5, 0.75, 1.2]
#     warm_up_period = 400
#     setting = SimulationSetting()
#     setting.runtime_type = RuntimeType.RELATIVE
#     setting.set_runtime(1.0)
#     setting.change_point_location_type = ChangePointLocationChoice.SPECIFIED
#     setting.set_change_point(1000)
#     setting.use_detection_window = True
#     detection_times = [50, 100, 300, 500, 1000, 2000]
#     selected_average_run_lengths = [25000, 100000,  150000]
#     batch_sizes = [10, 25, 50]
#     mu_baseline = 1
#     num_runs = 2000
#     pkl_directory = "./Results/GLRT_ROSS/ARL_0/"
#     h_selector = ThresholdSelector(pkl_directory)
#
#     log_directory = "./Results/GLRT_ROSS/Performance_Tests/"
#     log_file_name = log_directory + "run_length_test_log_"
#     data_store = pd.DataFrame(columns=['ARL_0', 'Batch Size', 'ARL_1', 'ARL_1_std', 'False Detection Rate',
#                                        'Mean Lag 1 AutoCorrelation', "Std Lag 1 AutoCorrelation",
#                                        'Conditional_Correct_Detection',
#                                        'Correct_Detection',
#                                        'Detection_Time'])
#     power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
#     detection_delay_df = simulate_detection_delays(batch_sizes, selected_average_run_lengths, rho,
#                                                    delta_rho_rel, mu_baseline, num_runs, h_selector, power_delay_log,
#                                                    setting, detection_windows=detection_times)
#     detection_delay_df.to_csv("./Results/GLRT_ROSS/Performance_Tests/GoldsmanResults_latest.csv")


def simulate_single_detection(selected_probability_of_false_detections, desired_autocorrelations, rho_vec,
                              delta_rho_rel, mu_baseline, num_runs, h_selector, power_delay_log, sim_setting,
                              data_store):
    """
        Simulate change point detection process with a single batch size test
        by picking the right threshold based on the desired probability of false detection
        and picking the batch size based off the desired autocorrelation
        :param selected_probability_of_false_detections: set of desired probability of false detections
        :param desired_autocorrelations: autocorrelation vector
        :param rho_vec: vector of intensity ratios
        :param delta_rho_rel: vector of relative increments in intensity ratio
        :param mu_baseline: baseline service rate
        :param num_runs: number of runs to play
        :param h_selector: Threshold functor for the detection test
        :param power_delay_log: object used to save the statistics of the test
        :param sim_setting: object of the class SimulationSetting
        :param data_store: df to store detection results
        :return: dataframe
    """
    average_run_lengths = [int(round(1 / fdp)) for fdp in selected_probability_of_false_detections]
    print("Simulating {} runs".format(num_runs))
    mu = mu_baseline
    start_time = 0
    my_service_rates = [mu, mu]
    if sim_setting.use_detection_window:
        print("There is a detection window")
        for rho in rho_vec:
            for delta_rho in delta_rho_rel:
                print("Running Detection Length Test")
                print("Current rho={0:2.4f}, delta rho={1:2.4f}".format(rho, delta_rho))
                arr_rate_0 = mu * rho
                data_store = simulate_detection_delay_by_rho_batch_on_auto_correlation(data_store,
                                                                                       desired_autocorrelations, rho,
                                                                                       delta_rho, arr_rate_0,
                                                                                       average_run_lengths, num_runs,
                                                                                       start_time,
                                                                                       sim_setting.runtime_num,
                                                                                       my_service_rates, h_selector,
                                                                                       sim_setting)
                power_delay_log.write_data(data_store)
    else:
        print("No detection window")
        for rho_idx, rho in enumerate(rho_vec):
            arr_rate_0 = mu * rho
            for delta_rho_idx, delta_rho in enumerate(delta_rho_rel):
                # end delta_rho
                print("Current rho={0:2.4f}, delta rho={1:2.4f}".format(rho, delta_rho))
                data_store = simulate_detection_delay_by_rho_batch_on_auto_correlation(data_store,
                                                                                       desired_autocorrelations, rho,
                                                                                       delta_rho, arr_rate_0,
                                                                                       average_run_lengths, num_runs,
                                                                                       start_time,
                                                                                       sim_setting.runtime_num,
                                                                                       my_service_rates, h_selector,
                                                                                       sim_setting)
                power_delay_log.write_data(data_store)
            # end rho
    return data_store


def main_3():
    """
    Change point detection based on a single batch.
    - The detection threshold of the GLRT is decided by the probability of false detection gamma ( 1  / ARL_0)
    - The batch size is the lowest batch size that achieves the desired autocorrelation
    """
    rho = [0.5]
    delta_rho_rel = [0.25, 0.5, 0.75, 1.2]
    setting = SimulationSetting()
    # Have the runtime be the ARL_0 length
    setting.runtime_type = RuntimeType.RELATIVE
    setting.set_runtime(1.0)
    setting.change_point_location_type = ChangePointLocationChoice.GEOMETRIC
    setting.set_change_point(200)
    change_point_location_vector = [200, 500, 1000, 2000, 5000, 10000]
    setting.use_detection_window = True
    mu_baseline = 1
    num_runs = 5000
    selected_probability_of_false_detections = [1e-5, 2e-4, 1e-4, 2e-3, 1e-3]
    selected_autocorrelations = [0.01, 0.05, 0.1]
    threshold_input_directory = "./Results/GLRT_ROSS/ARL_0/"
    h_selector = ThresholdSelector(threshold_input_directory)
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/"
    log_file_name = log_directory + "single_test_log_"
    data_store = pd.DataFrame()
    power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
    #     for change_point in change_point_location_vector:

    # setting.set_change_point(change_point)
    data_store = simulate_single_detection(selected_probability_of_false_detections,
                                           selected_autocorrelations,
                                           rho, delta_rho_rel, mu_baseline, num_runs, h_selector,
                                           power_delay_log, setting, data_store)
    data_store.to_csv("./Results/GLRT_ROSS/Performance_Tests/SingleDetection.csv", index=False)


def main_simple_roc():
    """
    Pick one batch size, but multiple thresholds
    """
    thresholds = np.linspace(0.01, 200, 50)
    rho = 0.5
    delta_rho = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.2]
    start_time = 0
    end_time = 1e5
    num_runs = 1000
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/Logs/"
    log_file_name = log_directory + "simple_single_test_log_"
    data_store = pd.DataFrame()
    mu = 1.0
    arr_rate_0 = mu * rho
    my_service_rates = [mu, mu]
    batch_size = 200
    power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
    data_store = simulate_detection_delay_by_rho_delta_rho_specific_batch_size(data_store, rho, delta_rho, arr_rate_0,
                                                                               num_runs, start_time, end_time,
                                                                               my_service_rates, thresholds,
                                                                               batch_size, power_delay_log)
    data_store.to_csv(f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_Batch_of_size_{batch_size}_long.csv",
                      index=False)


def main_wait_times_simple_rcpm(batch_size, is_parametric=True):
    """
    Run a change point detection for a specific batch size or for no batch size by Calling Gordon J. Ross CPM R-package
    2020-10-28, See the following for more details
            2015: ``Parametric and nonparametric sequential change detection in R''
    :param batch_size: The size of the batch used to aggregate the results. If batch_size=None, then run a nonparametric
    change point detection on the raw data with no batches.
    :return: None
    """
    rhos = [0.25, 0.5, 0.75]
    delta_rho = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.2]
    start_time = 0
    end_time = 1e5
    num_runs = 1000
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/Logs/"
    log_file_name = log_directory + "simple_single_rcpm_test_log_"
    data_store = pd.DataFrame()
    mu = 1.0
    power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
    # Personal Machine
    print(R_VERSION_BUILD)
    cpm = importr("cpm", lib_loc="C:/Users/swaho/OneDrive/Documents/R/win-library/3.6")
    # ATL-LAB
    # cpm = importr("cpm", lib_loc="C:/Users/hnikoue3/Documents/R/R-3.4.3/library")
    # cpm = importr("cpm")
    if is_parametric:
        with open('detectChangePointwithCPM.R', 'r') as f:
            r_string = f.read()
        cpm_func = STAP(r_string, "Detect_r_cpm_GaussianChangePoint")
    else:
        with open("detectChangePointwithCPM.R", 'r') as f:
            r_string = f.read()
        cpm_func = STAP(r_string, "Detect_r_cpm_NonParametricChangePoint")
    for rho in rhos:
        arr_rate_0 = mu * rho
        my_service_rates = [mu, mu]
        data_store, false_positive_list = simulate_changepoints_in_waitingtimes_using_r_cpm(data_store, rho, delta_rho,
                                                                                            arr_rate_0,
                                                                                            num_runs, start_time,
                                                                                            end_time,
                                                                                            my_service_rates,
                                                                                            batch_size,
                                                                                            power_delay_log, cpm_func)
        plot_distribution_of_false_positive_detection_times(false_positive_list, batch_size, is_parametric)
        if is_parametric:
            data_store.to_csv(
                f"./Results/GLRT_ROSS/Performance_Tests/Wait_Times/SimpleDetection_Batch_of_size_{batch_size}_rcpm.csv",
                index=False)
        else:
            if batch_size is None:
                data_store.to_csv(
                    "./Results/GLRT_ROSS/Performance_Tests/Wait_Times/SimpleDetection_Batch_of_size_nonparametric_rcpm.csv",
                    index=False)
            else:
                data_store.to_csv(
                    f"./Results/GLRT_ROSS/Performance_Tests/Wait_Times/SimpleDetection_Batch_of_size_{batch_size}_rcpm_nonparam.csv",
                    index=False)


def main_process_age_rcpm(batch_size, is_parametric=True):
    """
    Run a change point detection for a specific batch size or for no batch size by Calling Gordon J. Ross CPM R-package
    2020-10-28, See the following for more details
            2015: ``Parametric and nonparametric sequential change detection in R''
    :param batch_size: The size of the batch used to aggregate the results. If batch_size=None, then run a nonparametric
    change point detection on the raw data with no batches.
    :return: None
    """
    rhos = [0.25, 0.75]
    delta_rho = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.2]
    start_time = 0
    end_time = 1e5
    num_runs = 1000
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/Logs/"
    log_file_name = log_directory + "simple_single_rcpm_test_log_"
    data_store = pd.DataFrame()
    mu = 1.0
    power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
    print(R_VERSION_BUILD)
    cpm = importr("cpm", lib_loc="C:/Users/swaho/OneDrive/Documents/R/win-library/3.6")
    # cpm = importr("cpm")
    if is_parametric:
        with open('detectChangePointwithCPM.R', 'r') as f:
            r_string = f.read()
        cpm_func = STAP(r_string, "Detect_r_cpm_GaussianChangePoint")
    else:
        with open("detectChangePointwithCPM.R", 'r') as f:
            r_string = f.read()
        cpm_func = STAP(r_string, "Detect_r_cpm_NonParametricChangePoint")
    for rho in rhos:
        arr_rate_0 = mu * rho
        my_service_rates = [mu, mu]
        data_store, false_positive_list = simulate_change_points_in_age_process(data_store, rho, delta_rho, arr_rate_0,
                                                                                num_runs, start_time, end_time,
                                                                                my_service_rates, batch_size,
                                                                                power_delay_log, cpm_func,
                                                                                age_type="median")
        # plot_distribution_of_false_positive_detection_times(false_positive_list, batch_size, is_parametric)
        if is_parametric:
            data_store.to_csv(
                f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_age_of_process_Batch_of_size_{batch_size}_rho{rho * 100}_rcpm.csv",
                index=False)
        else:
            if batch_size is None:
                data_store.to_csv(
                    f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_age_of_process_Batch_of_size_nonparametric_rho{rho * 100}_rcpm.csv",
                    index=False)
            else:
                data_store.to_csv(
                    f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_age_of_process_Batch_of_size_{batch_size}_rho{rho * 100}_rcpm_nonparam.csv",
                    index=False)


def main_process_queue_length_rcpm(batch_size, is_parametric=True):
    """
    Run a change point detection for a specific batch size or for no batch size by Calling Gordon J. Ross CPM R-package
    2020-10-28, See the following for more details
            2015: ``Parametric and nonparametric sequential change detection in R''
    :param batch_size: The size of the batch used to aggregate the results. If batch_size=None, then run a nonparametric
    change point detection on the raw data with no batches.
    :return: None
    """
    rhos = [0.5]
    delta_rho = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.2]
    start_time = 0
    end_time = 1e5
    num_runs = 1000
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/Logs/"
    log_file_name = log_directory + "simple_single_rcpm_test_log_"
    data_store = pd.DataFrame()
    mu = 1.0
    power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
    #     cpm = importr("cpm")
    cpm = importr("cpm", lib_loc="C:/Users/swaho/OneDrive/Documents/R/win-library/3.6")
    if is_parametric:
        with open('detectChangePointwithCPM.R', 'r') as f:
            r_string = f.read()
        cpm_func = STAP(r_string, "Detect_r_cpm_GaussianChangePoint")
    else:
        with open("detectChangePointwithCPM.R", 'r') as f:
            r_string = f.read()
        cpm_func = STAP(r_string, "Detect_r_cpm_NonParametricChangePoint")
    for rho in rhos:
        arr_rate_0 = mu * rho
        my_service_rates = [mu, mu]
        data_store, false_positive_list = simulate_changepoints_in_queuelengths_using_r_cpm(data_store, rho, delta_rho,
                                                                                            arr_rate_0, num_runs,
                                                                                            start_time, end_time,
                                                                                            my_service_rates,
                                                                                            batch_size,
                                                                                            power_delay_log, cpm_func)
        if is_parametric:
            data_store.to_csv(
                f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_queue_length_Batch_of_size_{batch_size}_rho{rho * 100}_rcpm.csv",
                index=False)
        else:
            if batch_size is None:
                data_store.to_csv(
                    f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_queue_length_Batch_of_size_nonparametric_rho{rho * 100}_rcpm.csv",
                    index=False)
            else:
                data_store.to_csv(
                    f"./Results/GLRT_ROSS/Performance_Tests/SimpleDetection_queue_length_Batch_of_size_{batch_size}_rho{rho * 100}_rcpm_nonparam.csv",
                    index=False)


def main_process_joint_observations(batch_size):
    rhos = [0.25, 0.5, 0.75]
    delta_rho = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.2]
    start_time = 0
    end_time = 1e5
    num_runs = 1000
    log_directory = "./Results/GLRT_ROSS/Performance_Tests/Logs/"
    log_file_name = log_directory + "joint_tests_log_"
    data_store = pd.DataFrame()
    mu = 1.0
    power_delay_log = PowerTestLogger(log_file_name, is_full_path=False, file_type="bz2", dataframe=data_store)
    # Personal Machine
    print(R_VERSION_BUILD)
    cpm = importr("cpm", lib_loc="C:/Users/swaho/OneDrive/Documents/R/win-library/3.6")
    with open('detectChangePointwithCPM.R', 'r') as f:
        r_string = f.read()
    cpm_func = STAP(r_string, "Detect_r_cpm_GaussianChangePoint")
    for rho in rhos:
        arr_rate_0 = mu * rho
        my_service_rates = [mu, mu]
        #        data_store = simulate_joint_change_points_conditioned_on_hypothesis_outcome(data_store, rho, delta_rho, arr_rate_0, num_runs,
        #                                                                                    start_time, end_time, my_service_rates, batch_size, power_delay_log,
        #                                                                                    cpm_func)
        #        data_store.to_csv(
        #            "./Results/GLRT_ROSS/Performance_Tests/ChangePoint_Conditioned_on_HypothesisOutcome/" +
        #            "JointDetection_Batch_of_size_{}_rho{}_rcpm.csv".format(batch_size, rho * 100),
        #            index=False)

        data_store = simulate_joint_hypothesis_outcome_conditioned_on_change(data_store, rho, delta_rho, arr_rate_0,
                                                                             num_runs, start_time, end_time,
                                                                             my_service_rates, batch_size,
                                                                             power_delay_log, cpm_func)
        data_store.to_csv(
            "./Results/GLRT_ROSS/Performance_Tests/Hypothesis_Conditioned_on_Change/" +
            "JointDetection_Batch_of_size_{}_rho{}_rcpm.csv".format(batch_size, rho * 100),
            index=False)


if __name__ == "__main__":
    # Test with autocorrelation results
    # main_3()
    #     main_simple_roc()
    #     main_wait_times_simple_rcpm(50, is_parametric=False)
    #     main_wait_times_simple_rcpm(100, is_parametric=False)
    #     main_wait_times_simple_rcpm(150, is_parametric=False)
    #     main_wait_times_simple_rcpm(200, is_parametric=False)
    #     main_wait_times_simple_rcpm(500, is_parametric=False)
    #     main_wait_times_simple_rcpm(1000, is_parametric=False)
    #     main_process_age_rcpm(100)
    #     main_process_age_rcpm(200)
    #     main_process_age_rcpm(500)
    #     main_process_age_rcpm(1000, is_parametric=True)
    #     main_process_queue_length_rcpm(100)
    #     main_process_queue_length_rcpm(200)
    #     main_process_queue_length_rcpm(500)
    #     main_process_queue_length_rcpm(1000, is_parametric=True)
    #     main_process_age_rcpm(100)
    #     main_process_age_rcpm(200)
    #     main_process_age_rcpm(500)
    #     main_process_age_rcpm(1000, is_parametric=True)
    #     main_process_queue_length_rcpm(100)
    #     main_process_queue_length_rcpm(200)
    #     main_process_queue_length_rcpm(500)
    #     main_process_queue_length_rcpm(1000, is_parametric=True)
    main_process_joint_observations(100)
    main_process_joint_observations(200)
    main_process_joint_observations(500)
    main_process_joint_observations(1000)
    main_process_joint_observations(2000)
