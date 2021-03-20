"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: estimation_facility.py
Description: Various methods and classes used to quickly estimate distribution parameters
supports main_tests.py
Author: Harold Nemo Adodo Nikoue
part of my chapter on partial observability in my thesis
"""
import math
import warnings
from typing import List

import numpy as np
import scipy.stats
from scipy.special import digamma

from average_run_length_loader import return_ross_threshold

# from statsmodels.stats.stattools import durbin_watson
# import statsmodels.api as sm

np.seterr(all='warn')
warnings.filterwarnings('error')


class DetectionResults:
    def __init__(self):
        self.detection_index = float("inf")
        self.detection_value = 0
        self.change_point_index = float("inf")
        self.autocorrelation = 0
        self.db_stat = 0  # Durbin-Watson autocorrelation statisti


class DetectionStatistics:
    def __init__(self, detection_result: DetectionResults, actual_times: List[float]):
        self._detection_result = detection_result
        self._actual_times = actual_times  # batch_centers
        self.detection_time = float('inf')
        self.detection_value = 0
        self.change_point_time = float('inf')
        self.detection_delay = float('inf')
        self.false_alarm_length = 0
        self.autocorrelation = float('-inf')
        self.db_statistic = float('inf')  # Durbin-Watson statistics
        self._assign_statistics()

    def _assign_statistics(self):
        if not math.isinf(self._detection_result.detection_index):
            detection_time = self._actual_times[self._detection_result.detection_index]
            estimated_change_point_time = self._actual_times[self._detection_result.change_point_index]
            assert (estimated_change_point_time <= detection_time)
            self.detection_time = detection_time
            self.change_point_time = estimated_change_point_time
            self.detection_delay = (detection_time - estimated_change_point_time)
        self.detection_value = self._detection_result.detection_value
        self.autocorrelation = self._detection_result.autocorrelation
        self.db_statistic = self._detection_result.db_stat

    def return_detection_delay(self):
        return self.detection_delay


def compute_wait_times_threshold_m_m_1(significance: float, arrival_rate: float, service_rate: float):
    return -np.log(significance) / (service_rate - arrival_rate)


def compute_exponential_pdf(rate, x):
    return rate * np.exp(-rate * x)


def compute_gaussian_pdf(mean, std, x):
    return scipy.stats.norm.pdf(x, loc=mean, scale=std)


def estimate_mle_exponential(exponential_sum: float, length_observation: int) -> float:
    """
    Get MLE estimate of the exponential sum
    :param exponential_sum:
    :param length_observation:
    :return:
    """
    return length_observation / exponential_sum


def compute_exponential_correction_factor(k, n):
    k = k - 1
    return -2 * (k * digamma(k) + (n - k) * digamma(n - k) - n * digamma(n) + n * np.log(n) - k * np.log(k) - (n - k)
                 * np.log(n - k))


def compute_ross_gaussian_correction_factor(k, t):
    return t * (np.log(2 / t) + digamma((t - 1) / 2)) - k * (np.log(2 / k) + digamma((k - 1) / 2)) - (t - k) * (
            np.log(2 / (t - k)) +
            digamma((t - k - 1) / 2))


def compute_exponential_test_statistic(sum_n, sum_k_n, sum_0_k, window_start, k, n):
    test_statistic = 0
    k_prime = k - window_start
    try:
        test_statistic = 2 * (n - k_prime + 1) * np.log((n - k_prime + 1) / sum_k_n)
        test_statistic += 2 * (k_prime - 1) * np.log((k_prime - 1) / sum_0_k)
        test_statistic -= 2 * n * np.log(n / sum_n)
        test_statistic /= compute_exponential_correction_factor(k_prime, n)
    except Warning:
        print("Warning in compute_test_statistic: ", k - 1, n - k + 1, n / sum_n, (n - k + 1) / sum_k_n,
              (k - 1) / sum_0_k)
    return test_statistic


class GaussianStatistic:
    def __init__(self, mean, var, num_points):
        self.mean = mean
        self.variance = var
        self.n = num_points


def compute_forward_updated_gaussian_statistic(cur_stat: GaussianStatistic, new_point: float, num_points: int):
    """
    Update the variance and mean estimations
    with the new point
    """
    crt_mean = cur_stat.mean
    crt_var = cur_stat.variance
    new_mean = 1.0 / num_points * ((num_points - 1) * crt_mean + new_point)
    new_var = (num_points - 1) / num_points * (crt_var + crt_mean * crt_mean + new_mean * new_mean -
                                               2 * new_mean * crt_mean) + \
              1.0 / num_points * (new_point - new_mean) * (new_point - new_mean)
    return GaussianStatistic(new_mean, new_var, num_points)


def compute_backward_updated_gaussian_statistic(cur_stat: GaussianStatistic, rm_point: float):
    """
    Update the variance and the mean parameters for the detection statistics after the change.
    Due to the nature of GLRT, rm_point is removed from the computation of the statistics
    """
    crt_mean = cur_stat.mean
    crt_var = cur_stat.variance
    cur_n = cur_stat.n
    new_n = cur_n - 1
    new_mean = 1.0 / new_n * (cur_n * crt_mean - rm_point)
    new_var = cur_n / new_n * (crt_var + crt_mean * crt_mean + new_mean * new_mean - 2 * new_mean * crt_mean) \
              - (rm_point - new_mean) * (rm_point - new_mean) / new_n
    return GaussianStatistic(new_mean, new_var, new_n)


def compute_ross_updated_gaussian_test_statistic(before_change_stats: GaussianStatistic,
                                                 after_change_stats: GaussianStatistic, candidate_change_point,
                                                 s_0_n: float, k: int, n: int):
    """
    Compute Ross Gaussian test statistic with a new data point update
    For the before_change_stat we are adding a point to compute the mean and the variance
    For the after_change_stat, we are removing a point from the mean and variance computation
    """
    test_statistic_correction = compute_ross_gaussian_correction_factor(k, n)
    new_before_change_stats = compute_forward_updated_gaussian_statistic(before_change_stats, candidate_change_point, k)
    new_after_change_stats = compute_backward_updated_gaussian_statistic(after_change_stats, candidate_change_point)
    s_0_k = new_before_change_stats.variance
    s_k_n = new_after_change_stats.variance
    try:
        test_statistic = k * np.log((s_0_n / s_0_k)) + (n - k) * np.log(s_0_n / s_k_n)
    except RuntimeWarning as e:
        print("For values of s_0_n={}, s_0_k={} and s_k_n={},".format(s_0_n, s_0_k, s_k_n))
        print("\t obtained the ratios: r1={}, r2={} and failed!!!".format((s_0_n / s_0_k), (s_0_n / s_k_n)))
        print("\t k={} and n-k={}".format(new_before_change_stats.n, new_after_change_stats.n))
        raise e
    return 2.0 * test_statistic / test_statistic_correction, new_before_change_stats, new_after_change_stats, s_0_n


def compute_ross_gaussian_test_statistic(observations_after_k, observations_before_k, all_observations):
    k = len(observations_before_k)
    n = k + len(observations_after_k)
    test_statistic_correction = compute_ross_gaussian_correction_factor(k, n)
    # test_statistic_correction = 1
    s_0_k = np.var(observations_before_k, ddof=1)
    new_before_change_stats = GaussianStatistic(np.mean(observations_before_k), s_0_k, k)
    s_k_n = np.var(observations_after_k, ddof=1)
    new_after_change_stats = GaussianStatistic(np.mean(observations_after_k), s_k_n, len(observations_after_k))
    s_0_n = np.var(all_observations, ddof=1)
    try:
        test_statistic = k * np.log((s_0_n / s_0_k)) + (n - k) * np.log(s_0_n / s_k_n)
    except RuntimeWarning as e:
        print("For values of s_0_n={}, s_0_k={} and s_k_n={},".format(s_0_n, s_0_k, s_k_n))
        print("\t obtained the ratios: r1={}, r2={} and failed!!!".format((s_0_n / s_0_k), (s_0_n / s_k_n)))
        print("\t k={} and n-k={}".format(k, n - k))
        raise e
    return 2.0 * test_statistic / test_statistic_correction, new_before_change_stats, new_after_change_stats, s_0_n


def compute_gaussian_test_statistic(observations_after_k, observations_before_k):
    """
    Compute test statistic and use Bartlett correction
    """
    observations_before_k = np.array(observations_before_k)
    observations_after_k = np.array(observations_after_k)
    window_size = len(observations_before_k) + len(observations_after_k)
    mu_hat_0 = np.mean(observations_before_k)
    mu_hat_1 = np.mean(observations_after_k)
    sigma_hat_1 = np.var(observations_after_k, ddof=0)
    sigma_hat_0 = np.var(observations_before_k, ddof=0)
    test_statistic = 2 * window_size * (np.log(sigma_hat_0) - np.log(sigma_hat_1))
    test_statistic += np.sum(np.square(observations_before_k - mu_hat_0)) / sigma_hat_0
    test_statistic -= np.sum(np.square(observations_after_k - mu_hat_1)) / sigma_hat_1
    return test_statistic


def detect_recursive_gaussian_change_point(observations: List[float], sequence_start: int, sequence_end: int):
    """
    Compute highest test statistic and likely change point in interval

    D_n = max_k D_{k,n} \qquad 2 \leq k \leq t-2

    the question is whether the detection time is k or n: the actual time at which we detected the change.
    :param sequence_end:
    :param sequence_start:
    :param observations:
    :return: highest value of generalized likelihood ratio test statistic and corresponding index
    """
    length_run = sequence_end - sequence_start + 1
    g_n_max = float('-inf')
    best_k = 0
    # k should be greater than one to avoid computing the log of zero in the compute_test_Statistic formula
    is_first_variance_computation = True
    for k in range(sequence_start + 2, sequence_end - 3):
        if is_first_variance_computation:
            g_k_max, stat_0_k, stat_k_n, s_0_n = compute_ross_gaussian_test_statistic(observations[k + 1:sequence_end],
                                                                                      observations[
                                                                                      sequence_start:k + 1],
                                                                                      observations)
            is_first_variance_computation = False
        else:
            g_k_max, stat_0_k, stat_k_n, s_0_n = compute_ross_updated_gaussian_test_statistic(stat_0_k, stat_k_n,
                                                                                              observations[k], s_0_n, k,
                                                                                              sequence_end)
        if g_k_max > g_n_max:
            g_n_max = g_k_max
            best_k = k
    return g_n_max, best_k


def detect_change_point_in_window(observations: List[float], sequence_start: int, sequence_end: int,
                                  dist_type='exponential'):
    """
    Compute highest test statistic and likely change point in interval

    D_n = max_k D_{k,n} \qquad 2 \leq k \leq t-2

    the question is whether the detection time is k or n: the actual time at which we detected the change.
    :param sequence_end:
    :param sequence_start:
    :param observations:
    :param dist_type:
    :return: highest value of generalized likelihood ratio test statistic and corresponding index
    """
    length_run = sequence_end - sequence_start + 1
    g_n_max = float('-inf')
    best_k = 0
    # k should be greater than one to avoid computing the log of zero in the compute_test_Statistic formula
    is_first_variance_computation = True
    stat_0_k = None
    for k in range(sequence_start + 2, sequence_end - 3):
        # print(sequence_start, k, sequence_end, sum_before_k, whole_sum)
        if dist_type == "exponential":
            running_sums = np.cumsum(observations[sequence_start:sequence_end])
            idx_running_sum = k + 1 - sequence_start
            sum_before_k = running_sums[idx_running_sum]
            whole_sum = running_sums[-1]
            sum_after_k = whole_sum - sum_before_k
            g_k_max = compute_exponential_test_statistic(whole_sum, sum_after_k, sum_before_k, sequence_start, k,
                                                         length_run)
        elif dist_type == 'gaussian':
            # Start by computing the variance fully
            # and reset the variance after a large amount of change point detection
            if is_first_variance_computation or (k % 100 == 0):
                is_first_variance_computation = False
                try:
                    g_k_max, stat_0_k, stat_k_n, s_0_n = compute_ross_gaussian_test_statistic(
                        observations[k + 1:sequence_end], observations[sequence_start:k + 1], observations
                    )
                except RuntimeWarning:
                    print("\n Caught error thrown by compute_ross_updated_gaussian_test_statistic "
                          "in detect_change_point_in_window")
                    continue

            else:
                try:
                    g_k_max, stat_0_k, stat_k_n, s_0_n = compute_ross_updated_gaussian_test_statistic(stat_0_k,
                                                                                                      stat_k_n,
                                                                                                      observations[k],
                                                                                                      s_0_n, k,
                                                                                                      sequence_end)
                except RuntimeWarning:
                    print("\n Caught error thrown by compute_ross_updated_gaussian_test_statistic "
                          "in detect_change_point_in_window")
                    continue
        if g_k_max > g_n_max:
            g_n_max = g_k_max
            best_k = k
    return g_n_max, best_k


def detect_change_point_in_window_exponential(observations: List[float], sequence_start: int, sequence_end: int,
                                              dist_type='exponential'):
    """
    Compute highest test statistic and likely change point in interval

    D_n = max_k D_{k,n} \qquad 2 \leq k \leq t-2

    the question is whether the detection time is k or n: the actual time at which we detected the change.
    :param sequence_end:
    :param sequence_start:
    :param observations:
    :param window_size:
    :return: highest value of generalized likelihood ratio test statistic and corresponding index
    """
    length_run = sequence_end - sequence_start + 1
    running_sums = np.cumsum(observations[sequence_start:sequence_end])
    g_n_max = float('-inf')
    best_k = 0
    # k should be greater than one to avoid computing the log of zero in the compute_test_Statistic formula
    for k in range(sequence_start + 2, sequence_end - 2):
        idx_running_sum = k - sequence_start
        sum_before_k = running_sums[idx_running_sum]
        whole_sum = running_sums[-1]
        sum_after_k = whole_sum - sum_before_k
        # print(sequence_start, k, sequence_end, sum_before_k, whole_sum)
        g_k_max = compute_exponential_test_statistic(whole_sum, sum_after_k, sum_before_k, sequence_start, k,
                                                     length_run)
        if g_k_max > g_n_max:
            g_n_max = g_k_max
            best_k = k
    return g_n_max, best_k


def compute_wait_time_mean_rw(n, lambda_param, mu_param):
    """
    Compute the mean of the ladder process associated with the wait time
    """
    return n * (lambda_param - mu_param) / (lambda_param * mu_param)


def compute_wait_time_std_rw(n, lambda_param, mu_param):
    """
    Compute the standard deviation of the ladder process associated with the wait time
    """
    return np.sqrt(n * (1.0 / (lambda_param * lambda_param) + 1.0 / (mu_param * mu_param)))


class CuSumChangePointDetector:
    """
    Implement CuSum algorithm for exponential random variables
    :param null_params: parameters for the Null hypothesis passed as a tuple.
    """

    def __init__(self, null_params, alternative_params, threshold, dist_type='exponential'):
        self._distribution_type = dist_type
        if dist_type == "exponential":
            if isinstance(null_params, tuple):
                self._lambda_0 = null_params[0]
            else:
                self._lambda_0 = null_params
            if isinstance(alternative_params, tuple):
                self._lambda_1 = alternative_params[0]
            else:
                self._lambda_1 = alternative_params
        elif dist_type == "normal":
            self._mean_0 = null_params[0]
            self._sigma_0 = null_params[1]
            self._mean_1 = alternative_params[0]
            self._sigma_1 = alternative_params[1]
        elif dist_type == "normal2":
            self._lambda_0 = null_params[0]
            self._mu_0 = null_params[1]
            self._lambda_1 = alternative_params[0]
            self._mu_1 = alternative_params[1]
        self._performance_statistic = 0
        self._sequence_start = 0
        self._number_of_elements = 0
        self._running_sum = 0
        self._ht = threshold
        self._current_min_sum = float('inf')
        self._current_min_sum_index = None
        self.detection_results = []

    def update_exponential_cusum(self):
        if len(self._running_sum) > 0:
            detection_value_function = self._running_sum - self._current_min_sum
        else:
            detection_value_function = self._running_sum
        return max(detection_value_function, 0)

    def update_cusum(self, new_obs):
        """
        Update Cumulative sum
        :param new_obs: new observation
        :return: The value of the detection time and detection value if something has been detected
        """
        if self._distribution_type == "exponential":
            self._running_sum += np.log(compute_exponential_pdf(self._lambda_1, new_obs) /
                                        compute_exponential_pdf(self._lambda_0, new_obs))
            performance_statistic = self.update_exponential_cusum()

        elif "normal" in self._distribution_type:
            if self._distribution_type == "normal2" or self._distribution_type == "normal-no-variance":
                self._mean_1 = compute_wait_time_mean_rw(self._number_of_elements + 1, self._lambda_1, self._mu_1)
                self._mean_0 = compute_wait_time_mean_rw(self._number_of_elements + 1, self._lambda_0, self._mu_0)
                self._sigma_1 = compute_wait_time_std_rw(self._number_of_elements + 1, self._lambda_0, self._mu_0)
                self._sigma_0 = compute_wait_time_std_rw(self._number_of_elements + 1, self._lambda_0, self._mu_0)
            #             print("New obs: ", new_obs, "\t post-change mean: ", self._mean_1, "\t post-change variance: ",
            #                   self._sigma_1)
            #             print("New obs: ", new_obs, "\t pre-change mean: ", self._mean_0, "\t pre-change variance: ",
            #                   self._sigma_0)
            if self._distribution_type == "normal-no-variance":
                s_k = 1.0 / (2 * self._sigma_0 ** 2) * (2 * (self._mean_1 - self._mean_0) * new_obs + self._mean_0 ** 2
                                                        - self._mean_1 * self._mean_1)
            else:
                s_k = np.log(self._sigma_0) - np.log(self._sigma_1) \
                      + 1.0 / (2 * self._sigma_0 ** 2) * (new_obs - self._mean_0) ** 2 - 1.0 / (2 * self._sigma_1 ** 2) \
                      * (new_obs - self._mean_1) ** 2
            #             s_k = np.log(compute_gaussian_pdf(self._mean_1, self._sigma_1, new_obs) /
            #                          compute_gaussian_pdf(self._mean_0, self._sigma_0, new_obs))
            self._running_sum += s_k
            performance_statistic = max(self._performance_statistic + s_k, 0)

        if self._running_sum < self._current_min_sum:
            self._current_min_sum = self._running_sum
            self._current_min_sum_index = self._number_of_elements
        if performance_statistic >= self._ht:
            # print("CuSum found a change")
            detection_result = DetectionResults()
            # this will be used as an index not the length of the detection time
            detection_result.detection_index = self._number_of_elements
            detection_result.detection_value = performance_statistic
            detection_result.change_point_index = self._current_min_sum_index
            self.detection_results.append(detection_result)
            self._running_sum = 0
            self._current_min_sum_index = None
            self._current_min_sum = float('inf')
        self._number_of_elements += 1
        return performance_statistic

    def compute_change_point_locations(self, random_variates: List[float]) -> List[float]:
        """
        Sequential version of Cusum
        Return change points
        :param random_variates:
        :return:
        """
        detection_value_function_list = []
        for x_val in random_variates:
            detection_value_function_list.append(self.update_cusum(x_val))
        break_point_indices = []
        for detection_result in self.detection_results:
            break_point_indices.append(detection_result.change_point_index)
        return detection_value_function_list


class QueueThresholdTestDetector:
    def __init__(self, significance, arrival_rate, service_rate):
        self.sequence_start = 0
        self.detection_results = []
        self._ht = compute_wait_times_threshold_m_m_1(significance, arrival_rate, service_rate)

    def compute_change_point_locations(self, ladder_times: List[float], time_vector: List[float]) -> List[float]:
        """
        Sequential threshold test that just looks at the ladder times or even wait times to decide when
        some times are too large.
        :ladder_times: or wait-times of the queue
        :return: detection ladder times
        """
        detected_times = []
        detected_wait_times = []
        detected_indices = []
        for idx, ladder_time in enumerate(ladder_times):
            if ladder_time >= self._ht:
                detected_wait_times.append(ladder_time)
                detected_times.append(time_vector[idx])
                detected_indices.append(idx)
        return detected_indices, detected_times, detected_wait_times

    def get_threshold(self):
        return self._ht


def compute_lag_k_autocorrelation(array, lag=1):
    """
    Compute autocorrelation coefficient using corcoeff and recompute the mean
    """
    assert (len(array) > 1)
    if lag == 0:
        return 1
    else:
        return np.corrcoef(array[1:], array[:-1])[0][1]


class GLRTChangePointDetector:
    def __init__(self, threshold, gamma, dist_type='gaussian', use_Ross_threshold=True):
        # This start time is updated after each new change is detected
        self._sequence_start = 0
        self.detection_result = DetectionResults()
        self._gamma = gamma
        self._ht = threshold
        self._distribution_type = dist_type
        self._use_Ross_ht = use_Ross_threshold

    def compute_change_point_locations(self, random_variates: List[float]) -> List[float]:
        """
        Sequential change point algorithm by Gordon J. Ross
        Only detect a change point once and stops, no need to make multiple detections

        Return change points
        :param random_variates:
        :return:
        """
        #         break_points_times = []
        threshold = self._ht
        START_UP = 25
        # iterate through the different observations
        # we should start the sequence after 3 at least in accordance with G. J. Ross
        for n in range(self._sequence_start + START_UP, len(random_variates) - 3):
            g_n_max, break_point_index = detect_change_point_in_window(random_variates, self._sequence_start, n,
                                                                       self._distribution_type)
            # break_point_index is the index at which the change is believed to have first occurred.
            if self._use_Ross_ht and n > START_UP:
                threshold = return_ross_threshold(self._gamma, n)
            # ensure threshold is at least greater than 0.1 (arbitrary)
            #             threshold = max(0.01, threshold)
            if g_n_max > threshold:
                # we have detected a break_point_index
                #                print("Detected a change point at time={k} for a period ending at {n} with g_n_max={val}.".format(
                #                    k=times[break_point_index+1], n=times[n+1], val=g_n_max))
                #                 break_points_times.append(break_point_index)
                detection_result = DetectionResults()
                # the detection time is the end of the sequence that was being tested not k
                # detection_result.db_stat = durbin_watson(random_variates[self._sequence_start:n])
                autocorrelation_value, _ = scipy.stats.pearsonr(random_variates[self._sequence_start + 1:n],
                                                                random_variates[self._sequence_start:n - 1])
                detection_result.autocorrelation = autocorrelation_value
                #                 detection_result.autocorrelation = compute_lag_k_autocorrelation(
                #                     random_variates[self._sequence_start:n])
                detection_result.detection_index = n
                detection_result.detection_value = g_n_max
                detection_result.change_point_index = break_point_index
                self.detection_result = detection_result
                return threshold

        #             elif n > 50000 and (n % 50000 == 0):
        #                 # We have reached a high number of runs and restart the test to avoid a flat variance
        #                 self._sequence_start = n-3

        return threshold


def get_mle_exponential_mean(exponential_variates: List[float]) -> float:
    """
    Obtain the MLE estimator of the mean parameter for the exponential distribution
    :param exponential_variates: list of exponential random variables
    :return: lambda
    """
    return 1.0 / np.mean(exponential_variates)


def compute_test_statistic_exponential(exponential_variates: List[float], break_point: int) -> float:
    """
    Compute test statistic for an exponential according to
    Gordon J. Ross.  Sequential change detection in the presence of unknown parameters.Statistics and Computing,
    24(6):1017–1030, 2014
    :param exponential_variates:
    :param break_point: k in the algorithm
    :return: a value of the test statistic
    """
    n = len(exponential_variates)
    lambda_0_n = get_mle_exponential_mean(exponential_variates)
    lambda_post_break = get_mle_exponential_mean(exponential_variates[break_point:])
    lambda_pre_break = get_mle_exponential_mean(exponential_variates[:break_point])
    sum_n = sum(exponential_variates)
    sum_k_n = sum(exponential_variates[break_point:])
    sum_0_k = sum(exponential_variates[:break_point])
    test_statistic = -2 * np.log(lambda_0_n / lambda_pre_break) - 2 * (n - break_point) * np.log(
        lambda_0_n / lambda_post_break) + \
                     2 * lambda_0_n * sum_n - 2 * lambda_pre_break * sum_0_k - \
                     2 * lambda_post_break * sum_k_n
    return test_statistic
