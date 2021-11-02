"""

            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: online_parallel_test_config.py
Description:
    contains some general functions to support online_parallel_design_test_simulator.py
supports main_tests.py

Author: Harold Nemo Adodo Nikoue
end of my chapter on parallel partial observability in my thesis
Date: 10/25/2021
"""
from enum import Enum
from typing import List, Tuple

import numpy as np
import sklearn.metrics as metrics


class DetectionOutcome(Enum):
    FALSE_NOT_DETECTED = -2
    FALSE_EARLY_DETECTION = -1
    FALSE_DETECTION = 1
    CORRECT_NO_DETECTION = 1
    CORRECT_DETECTION = 2


class PolicyName(Enum):
    DETECT_ON_ALL_OBSERVATIONS = 0
    DETECT_ON_AGE = 3
    DETECT_ON_QUEUE = 4
    DETECT_ON_WAIT = 5


class ObservationType(Enum):
    AGE = 1
    QUEUE = 2
    WAIT = 3


class DetectionResult:
    hypothesis: str
    detection_likelihood: float
    detection_time: float
    batch_size: int
    observation_used: ObservationType

    def __init__(self, hypothesis: str, likelihood: float, detection_time: float, batch_size: int,
                 observation: ObservationType):
        self.hypothesis = hypothesis
        self.detection_likelihood = likelihood
        self.detection_time = detection_time
        self.batch_size = batch_size
        self.observation_used = observation


# set the parameters of the experiment
class ExperimentSetting:
    traffic_intensities: List[float]
    changes_in_traffic_intensities: List[float]
    batch_sizes: List[int]
    start_time: float
    end_time: float
    num_runs_per_config: int

    def __init__(self):
        self.traffic_intensities = [0.25, 0.5, 0.75]
        self.changes_in_traffic_intensities = [0.25, 0.5, 0.75, 1.0, 1.2]
        #        self.changes_in_traffic_intensities = [-0.25, 0, 0.25]
        self.batch_sizes = [100, 200, 300, 500, 1000, 1500, 2000]
        #        self.batch_sizes = [ 200, 300, 500]
        self.start_time = 0
        self.end_time = 1e4
        self.num_runs_per_config = 25


class SingleExperimentSetting:
    traffic_intensity: float
    change_in_traffic_intensity: float
    batch_sizes: List[int]
    start_time: float
    end_time: float
    num_runs_per_config: int
    policy_selected: PolicyName

    def __init__(self, rho, delta_rho, policy):
        self.traffic_intensity = rho
        self.change_in_traffic_intensity = delta_rho
        self.batch_sizes = [50, 100, 200, 300, 500, 1000, 1500, 2000]
        self.start_time = 0
        self.end_time = 1e4
        self.num_runs_per_config = 200
        self.policy_selected = policy


class TrackedStatisticsPerConfig:
    batch_size: int
    num_successes: int
    num_failures: int
    num_correct_detections: int
    num_correct_non_detections: int
    num_incorrect_non_detections: int
    num_incorrect_detections: int
    num_early_detections: int
    num_incorrect_detections_no_change: int
    likelihood_cd_vec: List[float]
    likelihood_fd_vec: List[float]
    detection_delays: List[float]
    policy: PolicyName

    def __init__(self, policy: PolicyName):
        self.policy = policy
        self.likelihood_cd_vec: List[float] = []
        self.likelihood_fd_vec: List[float] = []
        self.detection_delays: List[float] = []
        self.num_successes = 0
        self.num_failures = 0
        self.num_correct_detections = 0
        self.num_correct_non_detections = 0
        self.num_incorrect_non_detections = 0
        self.num_incorrect_detections = 0
        self.num_early_detections = 0
        self.num_incorrect_detections_no_change = 0
        self.batch_size = None

    def update_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def update_correct_detection(self, detection_delay, likelihood):
        self.likelihood_cd_vec.append(likelihood)
        self.detection_delays.append(detection_delay)
        self.num_correct_detections += 1
        self.num_successes += 1

    def update_correct_no_detection(self):
        self.num_successes += 1
        self.num_correct_non_detections += 1

    def update_false_detection(self, likelihood):
        self.likelihood_fd_vec.append(likelihood)
        self.num_failures += 1
        self.num_incorrect_detections += 1

    def update_early_detection(self, likelihood):
        self.likelihood_fd_vec.append(likelihood)
        self.num_failures += 1
        self.num_incorrect_detections += 1
        self.num_early_detections += 1

    def update_false_detection_no_change(self, likelihood):
        self.likelihood_fd_vec.append(likelihood)
        self.num_failures += 1
        self.num_incorrect_detections += 1
        self.num_incorrect_detections_no_change += 1

    def update_false_non_detection(self):
        self.num_failures += 1
        self.num_incorrect_non_detections += 1

    def return_avg_likelihood_cd(self):
        if len(self.likelihood_cd_vec) > 0:
            return np.mean(self.likelihood_cd_vec)
        return np.nan

    def return_avg_likelihood_fd(self):
        if len(self.likelihood_fd_vec):
            return np.mean(self.likelihood_fd_vec)
        else:
            return np.nan

    def return_y_rate(self):
        return (self.num_correct_detections + self.num_incorrect_detections) / (self.num_failures + self.num_successes)

    def return_avg_detection_delay(self):
        if len(self.detection_delays) > 0:
            return np.mean(self.detection_delays)
        else:
            return np.nan

    def return_cdr(self):
        return self.num_successes / float(self.num_successes + self.num_failures)

    def return_far(self):
        return self.num_incorrect_non_detections / float(self.num_successes + self.num_failures)

    def return_tp_rate(self):
        if (self.num_correct_detections + self.num_incorrect_non_detections) == 0:
            return 0
        return self.num_correct_detections / float(self.num_incorrect_non_detections + self.num_correct_detections)

    def return_fp_rate(self):
        if (self.num_incorrect_detections + self.num_correct_non_detections) == 0:
            return 0
        return self.num_incorrect_detections / float(self.num_correct_non_detections + self.num_incorrect_detections)

    def return_fp_rate_no_change(self):
        if (self.num_incorrect_detections_no_change + self.num_correct_non_detections) == 0:
            return 0
        return self.num_incorrect_detections_no_change / float(
            self.num_correct_non_detections + self.num_incorrect_detections)

    def return_fp_rate_change(self):
        if (self.num_early_detections + self.num_correct_non_detections) == 0:
            return 0
        return self.num_early_detections / float(self.num_correct_non_detections + self.num_incorrect_detections)

    def return_missed_detection_prob(self):
        if (self.num_incorrect_detections + self.num_correct_detections) == 0:
            return 0
        return self.num_incorrect_detections / float(self.num_incorrect_detections + self.num_correct_detections)

    def return_recall(self):
        return self.return_tp_rate()

    def return_sensitivity(self):
        return self.return_tp_rate()

    def return_precision(self):
        if (self.num_correct_detections + self.num_incorrect_detections) > 0:
            return self.num_correct_detections / float(self.num_correct_detections + self.num_incorrect_detections)
        else:
            return 0


def reorder_roc_lists(fpr_list: List[float], tpr_list: List[float]) -> Tuple[List[float], List[float]]:
    merged_list = [(fpr_list[idx], tpr_list[idx]) for idx in range(len(fpr_list))]
    print(merged_list)
    merged_list.sort(key=lambda x: x[0])
    new_fpr_list = []
    new_tpr_list = []
    for idx in range(len(merged_list)):
        new_fpr_list.append(merged_list[idx][0])
        new_tpr_list.append(merged_list[idx][1])
    return new_fpr_list, new_tpr_list


def return_roc_auc(fpr_list: List[float], tpr_list: List[float]):
    if len(fpr_list) < 2:
        return np.nan
    new_fpr_list, new_tpr_list = reorder_roc_lists(fpr_list, tpr_list)
    return metrics.auc(new_fpr_list, new_tpr_list)
