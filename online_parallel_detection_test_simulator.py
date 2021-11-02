"""

            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: online_parallel_detection_test_simulator.py
Description:
    Multiple detections of change points in a M/M/1 queue
    using the probabilities in joint_conditional_probability_hypothesis_conditioned_on_change.xlsx
supports main_tests.py

Test how early and with what accuracy a change can be detected
Author: Harold Nemo Adodo Nikoue
part of my chapter on parallel partial observability in my thesis
Date: 10/16/2021
"""
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from rpy2.robjects import FloatVector, r
from rpy2.robjects.packages import STAP, importr, SignatureTranslatedAnonymousPackage
from tqdm import tqdm

from batch_means_methods import create_nonoverlapping_batch_means
from generate_m_m_1_processes import simulate_deds_return_age_queue_wait
from online_parallel_test_config import PolicyName, ObservationType, DetectionResult, DetectionOutcome, \
    TrackedStatisticsPerConfig, SingleExperimentSetting, return_roc_auc, ExperimentSetting
from plotting_utility import generate_analysis_plots, generate_multiple_observation_roc_plots, \
    generate_multiple_observations_lift_curves, generate_multiple_observationss_cdr_vs_fa
from simulate_detection_delays import generate_random_change_point_time


class LikelihoodOfChangeDetector:
    _my_booster: xgb.Booster

    def __init__(self):
        xgboost_model_file_name = "condtional_change_xgboost_model.model"
        self._my_booster = xgb.Booster()
        self._my_booster.load_model(xgboost_model_file_name)

    def predict_from_df(self, df: pd.DataFrame) -> np.ndarray:
        return self._my_booster.predict(xgb.DMatrix(df[[
            "Batch Size",
            "rho",
            "delta_rho",
            "Run Length",
            "A+",
            "A-",
            "Q+",
            "Q-",
            "W+",
            "W-"
        ]].values))

    def predict_from_values(self, batch_size: int, rho: float, delta_rho: float,
                            a_plus: int, a_minus: int, q_plus: int, q_minus: int,
                            w_plus: int, w_minus: int) -> float:
        # form the dataframe
        input_dic = {
            "Batch Size": batch_size,
            "rho": rho,
            "delta_rho": delta_rho,
            "Run Length": 100000.,
            "A+": a_plus,
            "A-": a_minus,
            "Q+": q_plus,
            "Q-": q_minus,
            "W+": w_plus,
            "W-": w_minus
        }
        # need to be careful when working with a scalar dictionary
        df = pd.DataFrame(input_dic, index=[0])
        eval_dmatrix = xgb.DMatrix(df.values)
        return_array = self._my_booster.predict(eval_dmatrix)
        return return_array[0]


def detect_change_on_one_observation(batch_size: int, values: List[float], time_stamps: List[float],
                                     hypothesis: str, observation: ObservationType,
                                     cpm_func: SignatureTranslatedAnonymousPackage):
    batch_means, batch_centers = create_nonoverlapping_batch_means(values, time_stamps, batch_size=batch_size)
    rbatch_means = FloatVector(batch_means)
    r.assign('remote_batch_means', rbatch_means)
    r_estimated_changepoint_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_means)
    estimated_idx = r_estimated_changepoint_index[0] - 1
    if estimated_idx < 0:
        return None  # no detection
    detection_time = batch_centers[estimated_idx]
    return DetectionResult(hypothesis, np.nan, detection_time, batch_size, observation)


class CPDMostLikelyChangePolicy:
    """
    Policy that returns the smallest batch size/ detection delay change with
    a likelihood change above 0.95
    """
    _age_observations: List[float]
    _age_time_ts: List[float]
    _qlength_observations: List[float]
    _qlength_ts: List[float]
    _wait_times_observations: List[float]
    _wait_times_ts: List[float]
    _batch_sizes: List[int]
    _rho: float
    _delta_rho: float
    _cpm_func: SignatureTranslatedAnonymousPackage
    _likelihood_calculator: LikelihoodOfChangeDetector
    _acceptance_prob_threshold: float  # acceptance probability

    def __init__(
            self,
            rho: float, delta_rho: float, batch_sizes: List[int],
            age_observations: List[float],
            age_time_ts: List[float],
            queue_length_observations: List[float],
            queue_ts: List[float],
            wait_time_observations: List[float],
            wait_time_ts: List[float]
    ):
        self._age_observations = age_observations
        self._age_time_ts = age_time_ts
        self._qlength_observations = queue_length_observations
        self._qlength_ts = queue_ts
        self._wait_times_observations = wait_time_observations
        self._wait_times_ts = wait_time_ts
        self._rho = rho
        self._delta_rho = delta_rho
        self._batch_sizes = batch_sizes
        cpm = importr("cpm", lib_loc="C:/Users/swaho/OneDrive/Documents/R/win-library/3.6")
        with open('detectChangePointwithCPM.R', 'r') as f:
            r_string = f.read()
        self._cpm_func = STAP(r_string, "Detect_r_cpm_GaussianChangePoint")
        self._likelihood_calculator = LikelihoodOfChangeDetector()
        self._acceptance_prob_threshold = 0.95

    def detect_best_change(self, batch_sizes: List[int], policy_to_choose: PolicyName):
        """
        # Here I have a list of batch sizes to test then pick the smallest batch sizes
        # The results corresponds to the earliest detection at the desired likelihood of change threshold
        Return the return results as a tuple or None if no detection
        """
        best_detection_time = float('inf')
        best_likelihood = 0
        return_results = None
        for batch_size in sorted(batch_sizes):
            if policy_to_choose == PolicyName.DETECT_ON_ALL_OBSERVATIONS:
                detection_results = self.detect_a_change_on_3_observations(batch_size,
                                                                           self._age_observations, self._age_time_ts,
                                                                           self._qlength_observations, self._qlength_ts,
                                                                           self._wait_times_observations,
                                                                           self._wait_times_ts,
                                                                           self._cpm_func)
            elif policy_to_choose == PolicyName.DETECT_ON_AGE:
                detection_results = detect_change_on_one_observation(batch_size,
                                                                     self._age_observations, self._age_time_ts,
                                                                     "A+", ObservationType.AGE,
                                                                     self._cpm_func)
            elif policy_to_choose == PolicyName.DETECT_ON_QUEUE:
                detection_results = detect_change_on_one_observation(batch_size,
                                                                     self._qlength_observations, self._qlength_ts,
                                                                     "Q+", ObservationType.QUEUE,
                                                                     self._cpm_func)
            elif policy_to_choose == PolicyName.DETECT_ON_WAIT:
                detection_results = detect_change_on_one_observation(batch_size,
                                                                     self._wait_times_observations, self._wait_times_ts,
                                                                     "W+", ObservationType.WAIT,
                                                                     self._cpm_func)

            else:
                raise NotImplementedError
            if detection_results:
                detection_time = detection_results.detection_time
                if detection_time < best_detection_time:
                    return_results = detection_results
        #                    print("We got a good detection: ", detection_results)
        return return_results

    def detect_a_change_on_3_observations(self,
                                          batch_size: int, age_of_customers: List[float], age_times_ts: List[float],
                                          queue_lengths: List[float], queue_lengths_ts: List[float],
                                          wait_times: List[float],
                                          wait_times_ts: List[float],
                                          cpm_func: SignatureTranslatedAnonymousPackage):
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
        estimated_changepoint_age_idx = r_estimated_changepoint_age_index[0] - 1
        estimated_changepoint_queue_idx = r_estimated_changepoint_queue_index[0] - 1
        estimated_changepoint_wait_times_idx = r_estimated_changepoint_wait_times_index[0] - 1

        # So now we used the estimated change_point to decide where it is exactly
        # find the earliest detection

        # Start with the earliest detection
        # The batch indices should be the same since we have the same number of observations and same number of batches
        earliest_detection_idx, middle_detection_idx, last_detection_idx = sorted([estimated_changepoint_queue_idx,
                                                                                   estimated_changepoint_age_idx,
                                                                                   estimated_changepoint_wait_times_idx])

        if last_detection_idx < 0:
            # no detection
            return None
        # create the detection time
        if estimated_changepoint_age_idx > 0:
            age_detection_time = batch_centers_age[estimated_changepoint_age_idx]
        else:
            age_detection_time = float('inf')
        if estimated_changepoint_queue_idx > 0:
            queue_detection_time = batch_centers_queue[estimated_changepoint_queue_idx]
        else:
            queue_detection_time = float('inf')
        if estimated_changepoint_wait_times_idx > 0:
            wait_detection_time = batch_centers_wait[estimated_changepoint_wait_times_idx]
        else:
            wait_detection_time = float('inf')
        list_of_time_idx_tuples = [(estimated_changepoint_age_idx, 1),
                                   (estimated_changepoint_queue_idx, 2),
                                   (estimated_changepoint_wait_times_idx, 3)]
        # sort on the times
        times = {1: age_detection_time, 2: queue_detection_time, 3: wait_detection_time}
        earliest_detection_time_tup, middle_detection_time_tup, last_detection_time_tup = sorted(
            list_of_time_idx_tuples)
        earliest_detection_time = times[earliest_detection_time_tup[1]]
        middle_detection_time = times[middle_detection_time_tup[1]]
        last_detection_time = times[last_detection_time_tup[1]]
        # Start with earliest detection
        first_hypothesis = get_hypothesis_dic_at_idx(estimated_changepoint_age_idx, estimated_changepoint_queue_idx,
                                                     estimated_changepoint_wait_times_idx, earliest_detection_idx)
        first_detection_likelihood = self._likelihood_calculator.predict_from_values(batch_size, self._rho,
                                                                                     self._delta_rho,
                                                                                     first_hypothesis["A+"],
                                                                                     first_hypothesis["A-"],
                                                                                     first_hypothesis["Q+"],
                                                                                     first_hypothesis["Q-"],
                                                                                     first_hypothesis["W+"],
                                                                                     first_hypothesis["W-"]
                                                                                     )
        if earliest_detection_idx > 0 and first_detection_likelihood > self._acceptance_prob_threshold:
            return DetectionResult(first_hypothesis, first_detection_likelihood, earliest_detection_time, batch_size,
                                   ObservationType(earliest_detection_time_tup[1]))

        # Look at second detection
        second_hypothesis = get_hypothesis_dic_at_idx(estimated_changepoint_age_idx, estimated_changepoint_queue_idx,
                                                      estimated_changepoint_wait_times_idx, middle_detection_idx)
        second_detection_likelihood = self._likelihood_calculator.predict_from_values(batch_size, self._rho,
                                                                                      self._delta_rho,
                                                                                      second_hypothesis["A+"],
                                                                                      second_hypothesis["A-"],
                                                                                      second_hypothesis["Q+"],
                                                                                      second_hypothesis["Q-"],
                                                                                      second_hypothesis["W+"],
                                                                                      second_hypothesis["W-"]
                                                                                      )
        if middle_detection_idx > 0 and second_detection_likelihood > self._acceptance_prob_threshold:
            return DetectionResult(second_hypothesis, second_detection_likelihood, middle_detection_time, batch_size,
                                   ObservationType(middle_detection_time_tup[1]))

        # Third
        third_hypothesis = get_hypothesis_dic_at_idx(estimated_changepoint_age_idx, estimated_changepoint_queue_idx,
                                                     estimated_changepoint_wait_times_idx, last_detection_idx)
        third_detection_likelihood = self._likelihood_calculator.predict_from_values(batch_size, self._rho,
                                                                                     self._delta_rho,
                                                                                     third_hypothesis["A+"],
                                                                                     third_hypothesis["A-"],
                                                                                     third_hypothesis["Q+"],
                                                                                     third_hypothesis["Q-"],
                                                                                     third_hypothesis["W+"],
                                                                                     third_hypothesis["W-"]
                                                                                     )
        if last_detection_idx > 0:
            return DetectionResult(third_hypothesis, third_detection_likelihood, last_detection_time, batch_size,
                                   ObservationType(last_detection_time_tup[1]))
        else:
            return None


def get_hypothesis_dic_at_idx(age_detection_idx, queue_detection_idx, wait_detection_idx, current_idx):
    hypothesis_dic = {}
    if age_detection_idx <= current_idx:
        hypothesis_dic["A+"] = 1
        hypothesis_dic["A-"] = 0
    else:
        hypothesis_dic["A-"] = 1
        hypothesis_dic["A+"] = 0
    if queue_detection_idx <= current_idx:
        hypothesis_dic["Q+"] = 1
        hypothesis_dic["Q-"] = 0
    else:
        hypothesis_dic["Q-"] = 1
        hypothesis_dic["Q+"] = 0
    if wait_detection_idx <= current_idx:
        hypothesis_dic["W+"] = 1
        hypothesis_dic["W-"] = 0
    else:
        hypothesis_dic["W-"] = 1
        hypothesis_dic["W+"] = 0
    return hypothesis_dic


def get_detection_delays(time_of_detection: float, time_of_change: float) -> float:
    detection_delay = (time_of_detection - time_of_change) if (0 < time_of_detection < float('inf')) else np.nan
    return detection_delay


def simulate_and_detect_best_change_point(
        start_time: float, end_time: float,
        initial_arr_rate: float, final_arr_rate: float,
        service_rate: float,
        batch_sizes: List[float],
        policy_to_use: PolicyName
):
    disregard_frac = 0.05
    end_of_warmup_time = disregard_frac * end_time
    rho = initial_arr_rate
    delta_rho = (final_arr_rate - initial_arr_rate) / initial_arr_rate
    my_service_rates = [service_rate, service_rate]
    my_arrival_rates = [initial_arr_rate, final_arr_rate]
    if delta_rho < 1e-4:
        time_of_change = float('inf')
    else:
        time_of_change = generate_random_change_point_time(end_time, end_of_warmup_time)
    time_of_changes = [-1, time_of_change]
    queue_lengths, queue_lengths_ts, age_of_customers, age_times_ts, wait_times, wait_times_ts = \
        simulate_deds_return_age_queue_wait(start_time, end_time, my_arrival_rates,
                                            time_of_changes, my_service_rates)
    cpd_policy = CPDMostLikelyChangePolicy(rho, delta_rho, batch_sizes,
                                           age_of_customers, age_times_ts,
                                           wait_times, wait_times_ts,
                                           queue_lengths, queue_lengths_ts)

    detection_results = cpd_policy.detect_best_change(batch_sizes, policy_to_use)
    if detection_results:
        hypothesis = detection_results.hypothesis
        detection_likelihood = detection_results.detection_likelihood
        earliest_detection_time = detection_results.detection_time
        used_bs = detection_results.batch_size
        # The detection delay should be the same for all three observations
        # Check if it was a true change point:
        # There is no change  if time_of_change is inf
        if np.isinf(time_of_change):
            return DetectionOutcome.FALSE_DETECTION, np.nan, detection_likelihood, hypothesis, used_bs
        dd = get_detection_delays(earliest_detection_time, time_of_change)
        if np.isnan(dd) or dd < 0:
            # This is an early detection
            return DetectionOutcome.FALSE_EARLY_DETECTION, dd, detection_likelihood, hypothesis, used_bs
        else:
            # This is a correct detection
            return DetectionOutcome.CORRECT_DETECTION, dd, detection_likelihood, hypothesis, used_bs
    else:
        # NOTHING IS DETECTED
        if np.isinf(time_of_change):
            # detection delay of 0
            return DetectionOutcome.CORRECT_NO_DETECTION, 0.0, np.nan, None, None
        else:
            # the algorithm didn't detect any thing and there was a change
            return DetectionOutcome.FALSE_NOT_DETECTED, np.nan, np.nan, None, None


def run_experiments_and_collect_statistics(experiment_config: SingleExperimentSetting) -> TrackedStatisticsPerConfig:
    return_stat_object = TrackedStatisticsPerConfig(experiment_config.policy_selected)
    num_correct_detections = 0
    num_correct_non_detections = 0
    num_false_detections = 0
    num_false_non_detections = 0
    for _ in tqdm(range(experiment_config.num_runs_per_config), desc="Run per experiment progress"):
        new_arrival_rate = (1 + experiment_config.change_in_traffic_intensity) * experiment_config.traffic_intensity
        detection_outcome = simulate_and_detect_best_change_point(experiment_config.start_time,
                                                                  experiment_config.end_time,
                                                                  experiment_config.traffic_intensity,
                                                                  new_arrival_rate,
                                                                  1.0,
                                                                  experiment_config.batch_sizes,
                                                                  experiment_config.policy_selected
                                                                  )
        test_outcome, detection_delay, likelihood, chosen_hypothesis, best_batch_size = detection_outcome
        if best_batch_size:
            return_stat_object.update_batch_size(best_batch_size)
        if test_outcome == DetectionOutcome.CORRECT_DETECTION:
            return_stat_object.update_correct_detection(detection_delay, likelihood)
            num_correct_detections += 1
        #            print("Detection delay {:.3f}\n".format(detection_delay))
        elif test_outcome == DetectionOutcome.CORRECT_NO_DETECTION:
            return_stat_object.update_correct_no_detection()
            num_correct_non_detections += 1
        elif test_outcome == DetectionOutcome.FALSE_DETECTION:
            return_stat_object.update_false_detection_no_change(likelihood)
            num_false_detections += 1
        elif test_outcome == DetectionOutcome.FALSE_EARLY_DETECTION:
            return_stat_object.update_early_detection(likelihood)
            num_false_detections += 1
        elif test_outcome == DetectionOutcome.FALSE_NOT_DETECTED:
            return_stat_object.update_false_non_detection()
            num_false_non_detections += 1
        else:
            raise ValueError
    print("\n{0} TP, {1} FP, {2} FN, {3} TN".format(num_correct_detections, num_false_detections,
                                                    num_false_non_detections, num_correct_non_detections))
    return return_stat_object


def save_summary_statistics(dic_of_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
                            file_name: str) -> Tuple[float, float]:
    data_df = pd.DataFrame()
    tp_rate_list = []
    fp_rate_list = []
    fp_rate_list_no_change = []
    print("Saving output to df")
    for rho, inner_dic in dic_of_results.items():
        for delta_rho, tracked_stats in inner_dic.items():
            new_row_in_df = {
                "rho": rho,
                "delta_rho": delta_rho,
                "Batch Size": tracked_stats.batch_size,
                'ARL_1': tracked_stats.return_avg_detection_delay(),
                "Early Detections": tracked_stats.num_early_detections,
                'Missed Detection Prob': tracked_stats.return_missed_detection_prob(),
                "tp_rate": tracked_stats.return_tp_rate(),
                "fp_rate": tracked_stats.return_fp_rate(),
                "fp_rate_no_change": tracked_stats.return_fp_rate_no_change(),
                "fp_rate_change": tracked_stats.return_fp_rate_change(),
                'Correct_Detection': tracked_stats.return_cdr(),
                "Precision": tracked_stats.return_precision(),
                "Recall": tracked_stats.return_recall(),
                "TP": tracked_stats.num_correct_detections,
                "TN": tracked_stats.num_correct_non_detections,
                "FN": tracked_stats.num_incorrect_non_detections,
                "FP": tracked_stats.num_incorrect_detections,
                "YRate": tracked_stats.return_y_rate(),
                "Number Detections": tracked_stats.num_correct_detections + tracked_stats.num_incorrect_detections + \
                                     tracked_stats.num_correct_non_detections + \
                                     tracked_stats.num_incorrect_non_detections
            }
            fp_rate_list.append(tracked_stats.return_fp_rate())
            fp_rate_list_no_change.append(tracked_stats.return_fp_rate_no_change())
            tp_rate_list.append(tracked_stats.return_tp_rate())
            row_to_add = pd.Series(new_row_in_df)
            data_df = data_df.append(row_to_add, ignore_index=True)
    data_df.to_csv(file_name)
    auc_score = return_roc_auc(fp_rate_list, tp_rate_list)
    if not np.isnan(auc_score):
        print("The AUC score for the experiment is {:.2f}".format(auc_score))
    # tracking only the cases where there was no change sent to the model
    auc_score_no_change = return_roc_auc(fp_rate_list_no_change, tp_rate_list)
    if not np.isnan(auc_score_no_change):
        print("The AUC score_no_change for the experiment is {:.2f}".format(auc_score_no_change))

    return auc_score, auc_score_no_change


def run_experiment(policy_to_use: PolicyName) -> Dict[float, Dict[float, TrackedStatisticsPerConfig]]:
    general_configuration = ExperimentSetting()
    dic_of_results = {}
    for rho_idx in tqdm(range(len(general_configuration.traffic_intensities)), desc="rho iteration progress"):
        rho = general_configuration.traffic_intensities[rho_idx]
        dic_of_results[rho] = {}
        for delta_rho_idx in tqdm(range(len(general_configuration.changes_in_traffic_intensities)),
                                  desc=f"rho={rho} ({rho_idx}) drho iteration progress"):
            delta_rho = general_configuration.changes_in_traffic_intensities[delta_rho_idx]
            all_observations_config = SingleExperimentSetting(rho, delta_rho, policy_to_use)
            return_stat = run_experiments_and_collect_statistics(all_observations_config)
            return_stat.policy = policy_to_use
            dic_of_results[rho][delta_rho] = return_stat
            print("Mean detection delay: {}".format(return_stat.return_avg_detection_delay()))

    return dic_of_results


def manage_experiments_and_visualize():
    # configure the sim
    # iterate over rho and delta_rho
    policy_all = PolicyName.DETECT_ON_ALL_OBSERVATIONS
    dic_of_results_all = run_experiment(policy_all)
    auc_all, auc_all_no_change = save_summary_statistics(dic_of_results_all,
                                                         "Results/FinalResults_All_Observations.csv")
    generate_analysis_plots(dic_of_results_all, auc_all, auc_all_no_change, PolicyName.DETECT_ON_ALL_OBSERVATIONS)
    dic_of_results_age = run_experiment(PolicyName.DETECT_ON_AGE)
    auc_age, auc_age_no_change = save_summary_statistics(dic_of_results_age, "Results/FinalResults_Age.csv")
    dic_of_results_queue = run_experiment(PolicyName.DETECT_ON_QUEUE)
    auc_queue, auc_queue_no_change = save_summary_statistics(dic_of_results_queue, "Results/FinalResults_Queue.csv")
    dic_of_results_wait = run_experiment(PolicyName.DETECT_ON_WAIT)
    auc_wait, auc_wait_no_change = save_summary_statistics(dic_of_results_wait, "Results/FinalResults_Wait.csv")
    generate_multiple_observation_roc_plots(
        dic_of_results_all, auc_all, auc_all_no_change,
        dic_of_results_age, auc_age, auc_age_no_change,
        dic_of_results_queue, auc_queue, auc_queue_no_change,
        dic_of_results_wait, auc_wait, auc_wait_no_change
    )
    generate_multiple_observations_lift_curves(dic_of_results_all,
                                               dic_of_results_age, dic_of_results_queue, dic_of_results_wait)
    generate_multiple_observationss_cdr_vs_fa(dic_of_results_all, dic_of_results_age, dic_of_results_queue,
                                              dic_of_results_wait)
    return None


def main():
    manage_experiments_and_visualize()


if __name__ == "__main__":
    main()
