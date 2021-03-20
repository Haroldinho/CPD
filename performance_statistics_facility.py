"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: performance_statistics_facility.py
Description: Library to compute Average Run Length statistics and other statistics to evaluate the performance
of the change point detector
Author: Harold Nemo Adodo Nikoue
part of my partial observability thesis
"""

from collections import defaultdict
from typing import List

import numpy as np


def return_closest_number_in_vector(target_list, point):
    for i in range(len(target_list) - 1):
        if target_list[i] <= point <= target_list[i + 1]:
            if (target_list[i + 1] - point) > (point - target_list[i]):
                return target_list[i]
            else:
                return target_list[i + 1]
    return target_list[-1]


def return_index_closest_number_in_vector(target_list, point):
    for i in range(len(target_list) - 1):
        if target_list[i] <= point <= target_list[i + 1]:
            if (target_list[i + 1] - point) > (point - target_list[i]):
                return i
            else:
                return i + 1
    return len(target_list) - 1


class PowerTestStructure:
    """
    Class used to store a data structure for the power test statistics to find the right combination of detection
    threshold and batch sizes that  gives us the best detection delays.
    The batch size is not one of the argument of the matrix
    """

    def __init__(self, alphas):
        self._alphas = alphas
        self._inner_ds = {i: defaultdict(dict) for i in range(len(self._alphas))}
        self._min_alpha = float('-inf')
        self._max_alpha = float('inf')
        self._max_ht = float('inf')
        self._min_ht = float('-inf')

    def insert(self, alpha, h_t, value):
        alpha_idx = return_index_closest_number_in_vector(self._alphas, alpha)
        if alpha_idx in self._inner_ds and self._inner_ds[alpha_idx]:
            if h_t not in self._inner_ds[alpha_idx]:
                self._inner_ds[alpha_idx][h_t] = [value]
            else:
                self._inner_ds[alpha_idx][h_t].append(value)
        else:
            self._inner_ds = {alpha_idx: {h_t: [value]}}

    def return_dict(self):
        return self._inner_ds

    def return_matrix(self):
        num_alpha_col = self._max_alpha - self._min_alpha + 1
        num_ht_row = self._max_ht - self._min_ht + 1
        mat = np.zeros((num_ht_row, num_alpha_col))
        for alpha, h_t_struct in self._inner_ds.items():
            for h_t, val_list in h_t_struct.items():
                mat[alpha, h_t] = np.mean(val_list)
        return mat


class AverageRunLength:
    """
    Class that wraps around the different methods to compute average run lengths 0 and 1
    and make it easier to communicate data between those methods and return results
    """

    def __init__(self, true_change_points: List[float], detected_change_points: List[float]):
        self.marked_changed_points = []
        self._detected_changed_points = detected_change_points
        self._true_change_points = sorted(true_change_points)
        self.detection_times = []
        self.true_change_times_before_detection = []
        self.detection_delays = []
        self._compute_marked_changes()
        # mark all points as correct or incorrect detections

    def return_correct_detection_prob(self):
        return len(self.detection_times) / float(len(self._detected_changed_points))

    def compute_significance_level(self):
        # TODO
        return None

    def _compute_marked_changes(self) -> None:
        """
           Evaluate all detected change points as either correct or incorrect.
           For each candidate change point find the closest true change point smaller than the detection.
           if there is none it is a false detection
           if the closest change point has already been detected it is a false detection
           otherwise if the closest change point has not been detected, it is a correct detection.
           :param self:
           :return: a list of boolean of the same size as detected change points 0: if incorrect, 1 if correct.

           TO test this code just check that the number of marked detections correspond to the size of the detected
           change points
        """
        is_change_detected = [False for _ in self._true_change_points]
        mark_points_correctness = []
        for detector_idx in range(len(self._detected_changed_points)):
            detection_time = self._detected_changed_points[detector_idx]
            list_candidate_changes = [t for t in self._true_change_points if t <= detection_time]
            if len(list_candidate_changes) == 0:
                # this is a false detection
                mark_points_correctness.append(False)
            else:
                true_change_idx = len(list_candidate_changes) - 1
                if is_change_detected[true_change_idx]:
                    mark_points_correctness.append(False)
                else:
                    mark_points_correctness.append(True)
                    is_change_detected[true_change_idx] = True
                    self.detection_times.append(detection_time)
                    self.true_change_times_before_detection.append(list_candidate_changes[-1])
        self.marked_changed_points = mark_points_correctness

    def compute_arl_0(self):
        """
        Compute the average run length between two false positives
        To detect false positives I don't need to compare the marked change points to the truth.
        I just need to compare the incorrect detections two by two
        ASSUME detected_change_points are in increasing time order
        :param self:
        :return:
        Could test that the detected_change_points are indeed in increasing order
        TODO: Make a test for this method
        """
        # get the indices of all mis detected changes.
        indices_missed_detections = [i for i in range(len(self.marked_changed_points))
                                     if (self.marked_changed_points[i] == 0)]
        # iterate through these indices and compute the time difference compared to the previous time stamp in the set
        # and append to a list of false positive run length
        false_positive_run_time = []
        for idx in range(1, len(indices_missed_detections)):
            idx_current_missed_detection = indices_missed_detections[idx]
            idx_prev_missed_detection = indices_missed_detections[idx - 1]
            false_positive_run_time.append(self._detected_changed_points[idx_current_missed_detection]
                                           - self._detected_changed_points[idx_prev_missed_detection])
        if not false_positive_run_time:
            return 0.0
        # return the average
        return np.mean(false_positive_run_time)

    def compute_arl_1(self):
        """
        Compute the average run length between a true change point and its detection
        ASSUME detected_change_points and true_change_points are in increasing time order
        :param self:
        :return:
        """
        detection_delays = []
        for idx in range(len(self.detection_times)):
            detection_delays.append(self.detection_times[idx] - self.true_change_times_before_detection[idx])
        self.detection_delays = detection_delays
        if not detection_delays:
            return 0.0
        return np.mean(detection_delays)


def find_detection_index(detection_times, change_point_time):
    # verify that the detection times are pre-sorted
    assert (all(detection_times[i] <= detection_times[i + 1] for i in range(len(detection_times) - 1)))
    # if there are no detection times, don't return 0, return a very high number
    for detection_idx, detection_time in enumerate(detection_times):
        if detection_time >= change_point_time:
            return detection_idx
    return min(len(detection_times), float('inf'))
