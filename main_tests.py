"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: main_tests.py
Description: Use change point detection on  stream of exponential random variables which can either represent
    arrival times, departure times or service times

# 02/26/2020: After meeting with Dave Goldsman
Test three cases:
    1) Performing change point detection on non-homogeneous Exponentially distributed service times believed
    to be exponential with unknown parameters
    2) performing change point detection on non-homogeneous Weibull distributed service times believed to be
    exponential with unknown parameters
    3) Analyze the cost of false detection under these assumptions:
        a- What is the cost closing a server?
        b- What is the cost of opening a new one?
        c- What is the impact on the schedule?
        d- How much does it cost to detect the change
# 4/8/2020: After meeting with D. Goldsman
    Use Non-Overlapping Batch means to transform wait times into normal i.i.d. variables
    First verify the independent and normal assumptions
    Created test_wait_times_batch_means in response
THis code employs the unknown-parameter change point detection model developed by Hawkins and Zamba in 2012
"Statistical process control for shifts in mean or variance usinga changepoint formulation"
Gordon J. Ross in 2012 "Sequential change detection in the presence of unknown parameters.
Statistics and Computing" extended the procedure to sequential detection and exponential distribution.

Evaluation metrics: Detection delay, false alarm rates, ...
Author: Harold Nemo Adodo Nikoue
part of the partial observability thesis
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from batch_means_methods import create_nonoverlapping_batch_means
from estimation_facility import GLRTChangePointDetector, CuSumChangePointDetector, DetectionStatistics, \
    QueueThresholdTestDetector
from generate_m_m_1_processes import LadderPointQueue, SingleServerMarkovianQueue
from generate_stochastic_streams import generate_non_homogeneous_expo_variables
from performance_statistics_facility import AverageRunLength
from visualization import plot_wait_times, plot_score_vs_n, plot_wait_times_with_detected_changes, \
    plot_wait_times_against_D_k_n


# ATTENTION THE ORDER IN WHICH YOU IMPORT STATSMODELS MATTER


def set_up_logging():
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.INFO if you want to change the amount of output
    logger = logging.getLogger()
    # logger.setLevel(logging.WARN)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('./logs/warnings.log')
    fh.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class DetectionScoreRecorder:
    """
    Class that record G_n for different values of detection time n, and estimated change point k_hat
    """

    def __init__(self):
        self._detection_scores = []
        self._detection_times = []
        self._estimated_change_points = []

    def update_stat(self, g_n, k, n):
        self._detection_scores.append(g_n)
        self._detection_times.append(n)
        self._estimated_change_points.append(k)

    def update_stats(self, detection_stats: DetectionStatistics):
        self._detection_scores.extend(detection_stats.detection_values)
        self._detection_times.extend(detection_stats.detection_times)
        self._estimated_change_points.extend(detection_stats.change_point_times)

    def get_change_points(self):
        return self._estimated_change_points

    def get_detection_times(self):
        return self._detection_times

    def get_detection_scores(self):
        return self._detection_scores


def run_expo_expo_test():
    """
    Performing change point detection on non-homogeneous Exponentially distributed service times believed
    to be exponential with unknown parameters
    plot or print out the statistics for false detection
    :return: None
    """
    # my_list_rates = np.random.uniform(4, 20, 10)
    my_list_rates = [4, 20]
    time_of_changes = [0, 10]
    interval_length = 300
    num_sims = 5
    glrt_detection_times = []
    cusum_detection_times = []
    glrt_detection_perfs = DetectionScoreRecorder()
    cusum_detection_perfs = DetectionScoreRecorder()
    for num_iter in range(num_sims):
        # 1. generate a sequence of arrivals with some breaks or changes in arrival rates.
        event_times, actual_change_times = generate_non_homogeneous_expo_variables(my_list_rates, interval_length,
                                                                                   time_of_changes)

        # 2. Use Ross' change point detection algorithm to detect the changes.
        # convert to inter-event times which are exponentially distributed
        exponential_inter_event_times = []
        for i in range(1, len(event_times)):
            exponential_inter_event_times.append(event_times[i] - event_times[i - 1])

        print("Arrival times: ", event_times)
        print("Inter-arrival times: ", exponential_inter_event_times)
        # 3. Return the statistics on the number of changes detected, number of changes not detected
        #                               false alarm rate (number of false alarms divided by the number of detections)
        #                               correct detection rate (1-false alarm rate)
        #                               detection delay ( time it took to correctly detect a shift)
        #                               detection delays for positive and negative shifts
        cusum_threshold_t = 0.5
        cusum_change_point_estimator = CuSumChangePointDetector(my_list_rates[0], my_list_rates[1], cusum_threshold_t)
        g_n_cusum = cusum_change_point_estimator.compute_change_point_locations(exponential_inter_event_times.copy())
        cusum_detection_statistics = DetectionStatistics(cusum_change_point_estimator.detection_results, event_times)
        cusum_detected_change_times = cusum_detection_statistics.change_point_times
        cusum_detection_perfs.update_stats(cusum_detection_statistics)
        plt.plot(g_n_cusum)
        plt.xlabel('n')
        plt.ylabel('G_n')
        plt.title("CuSum Detection")
        plt.show()

        glrt_threshold_t = 8.2
        glrt_change_point_estimator = GLRTChangePointDetector(glrt_threshold_t, None)
        g_n_glrt = glrt_change_point_estimator.compute_change_point_locations(exponential_inter_event_times.copy())
        glrt_detection_statistics = DetectionStatistics(glrt_change_point_estimator.detection_result, event_times)
        glrt_detected_change_times = glrt_detection_statistics.change_point_times
        glrt_detection_perfs.update_stats(glrt_detection_statistics)
        plt.plot(g_n_glrt)
        plt.xlabel('n')
        plt.ylabel('G_n')
        plt.title("GLRT Detection")
        plt.show()

        print("Original changes at times: ", actual_change_times)
        print("GLRT Detected changes at times: ", glrt_detected_change_times)
        print("CuSum Detected changes at times: ", cusum_detected_change_times)
        # 3.b save for post Monte-Carlo analysis
        if glrt_detected_change_times:
            detected_change_points = glrt_detected_change_times
            if isinstance(detected_change_points, list):
                glrt_detection_times.extend(detected_change_points)
            else:
                glrt_detection_times.append(detected_change_points)
        if cusum_detected_change_times:
            detected_change_points = cusum_detected_change_times
            if isinstance(detected_change_points, list):
                cusum_detection_times.extend(detected_change_points)
            else:
                cusum_detection_times.append(detected_change_points)
        # 4. Compute the different statistics
        glrt_arl = AverageRunLength(time_of_changes[1:], glrt_detected_change_times)
        glrt_arl_0 = glrt_arl.compute_arl_0()
        glrt_arl_1 = glrt_arl.compute_arl_1()
        print("GLRT For h={hna}, we obtained arl_0={arl0} and arl_1={arl1}".format(hna=glrt_threshold_t,
                                                                                   arl0=glrt_arl_0,
                                                                                   arl1=glrt_arl_1))

        cusum_arl = AverageRunLength(time_of_changes[1:], cusum_detected_change_times)
        cusum_arl_0 = cusum_arl.compute_arl_0()
        cusum_arl_1 = cusum_arl.compute_arl_1()
        print("CuSum For h={hna}, we obtained arl_0={arl0} and arl_1={arl1}".format(hna=cusum_threshold_t,
                                                                                    arl0=cusum_arl_0,
                                                                                    arl1=cusum_arl_1))
    print("\n\nGLRT Mean detection points: {mean} and standard deviation: {std}".format(
        mean=np.mean(glrt_detection_times), std=np.std(glrt_detection_times)))
    print("\n\nCuSum Mean detection points: {mean} and standard deviation: {std}".format(
        mean=np.mean(cusum_detection_times), std=np.std(cusum_detection_times)))
    # plot Score vs detection time, and score vs chane point time
    glrt_all_detection_scores = glrt_detection_perfs.get_detection_scores()
    glrt_all_detection_times = glrt_detection_perfs.get_detection_times()
    glrt_all_k_hat = glrt_detection_perfs.get_change_points()
    plot_score_vs_n(glrt_all_detection_times, glrt_all_detection_scores, 'n', 'G_n', 'GLRT',
                    'Figures/glrt_g_n_vs_n.eps')
    plot_score_vs_n(glrt_all_k_hat, glrt_all_detection_scores, 'k_hat', 'G_n', 'GLRT', 'Figures/glrt_g_n_vs_k_hat.eps')

    cusum_all_detection_scores = cusum_detection_perfs.get_detection_scores()
    cusum_all_detection_times = cusum_detection_perfs.get_detection_times()
    cusum_all_k_hat = cusum_detection_perfs.get_change_points()
    plot_score_vs_n(cusum_all_detection_times, cusum_all_detection_scores, 'n', 'G_n', 'CuSum',
                    'Figures/cusum_g_n_vs_n.eps')
    plot_score_vs_n(cusum_all_k_hat, cusum_all_detection_scores, 'k_hat', 'G_n', 'CuSum',
                    'Figures/cusum_g_n_vs_k_hat.eps')

    # 5. Plot some of the output (ROC, Neyman-Pearson Curve)


def run_m_m_1_test():
    """
    Performing change point detection for a M/M/1 queue
    :return: None
    """
    start_time = 0
    end_time = 1000
    my_service_rates = [10, 10]
    my_arrival_rates = [4, 9]
    time_of_changes = [0, 100]
    interval_length = 300

    # One simulation of the ladder point process
    # Step 1: Simulate the ladder point process  (Create a class that inherits from LadderPointQueue
    ladder_point_creator = LadderPointQueue(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                            time_of_changes)
    wait_times = ladder_point_creator.simulate_ladder_point_process()
    ladder_times = ladder_point_creator.get_ladder_times()
    time_vector = ladder_point_creator.get_arrival_times()
    assert (len(ladder_times) == len(time_vector))
    # Step 2: Run simple threshold
    #           Collect the times at which changes were detected
    #           Classify each detection as a true detection or a false detection
    #           Compute the delay for each true detection
    alpha = 0.00001
    baseline_arrival_rate = my_arrival_rates[0]
    baseline_service_rate = my_service_rates[0]
    change_point_detector = QueueThresholdTestDetector(significance=alpha, arrival_rate=baseline_arrival_rate,
                                                       service_rate=baseline_service_rate)
    detected_change_idx, detected_change_times, detected_wait_times \
        = change_point_detector.compute_change_point_locations(ladder_times, time_vector)
    threshold = change_point_detector.get_threshold()

    # Step 3: Compute Average Run Lengths
    arl = AverageRunLength(time_of_changes[1:], detected_change_times)
    arl_0 = arl.compute_arl_0()
    arl_1 = arl.compute_arl_1()

    # Step 4: Aggregate, plot and summarize results
    #           Plot time series of wait times and peak detection times on top
    plot_wait_times_with_detected_changes(ladder_times, time_vector, detected_change_idx, threshold, time_of_changes[1],
                                          "wait_times_detected_{}_{}_Test1.png".format(
                                              my_arrival_rates[0], my_arrival_rates[1]))
    num_false_positive_change_time = [detected_time for detected_time in detected_change_times
                                      if detected_time < time_of_changes[1]]
    num_false_positives = len(num_false_positive_change_time)
    num_true_positives = len(detected_change_times) - num_false_positives
    total_number_time_steps_before_changepoint = len([time for time in time_vector if time < time_of_changes[1]])
    total_number_time_steps_after_changepoint = len(time_vector) - total_number_time_steps_before_changepoint
    false_positive_rate = float(num_false_positives) / total_number_time_steps_before_changepoint
    true_positive_rate = float(num_true_positives) / total_number_time_steps_after_changepoint
    print("The rate of false positive is: {} compared to {} for true positives.".format(false_positive_rate,
                                                                                        true_positive_rate))


def visualize_m_m_1_ladder_points():
    """
    Simulate a M/M/1 queue and visualize the random walks, the wait times and idle times
    """
    start_time = 0
    end_time = 1000
    arrival_rate = 9.0
    service_rate = 10.0
    arrival_change_points = [0]
    service_change_points = [0]
    ladder_point_creator = LadderPointQueue(start_time, end_time, [arrival_rate],
                                            arrival_change_points, [service_rate],
                                            service_change_points)
    wait_times = ladder_point_creator.simulate_ladder_point_process()
    random_walk_points = ladder_point_creator.get_ladder_times()

    plt.savefig("S_n_vs_n_ladderprocess.jpg")
    plot_wait_times(wait_times, arrival_rate, service_rate, "Figures/W_n_vs_n_ladderprocess.jpg")


def visualize_m_m_1_deds():
    """
    Simulate a M/M/1 queue and visualize the random walks, the wait times and idle times
    """
    start_time = 0
    end_time = 1000
    arrival_rate = 4.0
    service_rate = 10.0
    arrival_change_points = [0]
    service_change_points = [0]
    deds_creator = SingleServerMarkovianQueue(start_time, end_time, [arrival_rate],
                                              arrival_change_points, [service_rate],
                                              service_change_points)
    wait_times = deds_creator.simulate_deds_process()
    plot_wait_times(wait_times, arrival_rate, service_rate, "W_n_vs_n_deds_process.jpg")


def run_detection_tests_on_independent_wait_times():
    """
    Performs CuSum and GLRT on wait times after having reduced the correlation by using Batch means
    1. Generate wait times for a M/M/1 queue for a time period [0,100] with a change of parameters at t=50
         from (\lambda=9, \mu=10) to (\lambda=4, \mu=10)
    2. Get the means of 6-10 different batches and return the time stamp of the center of the batch
    3. Run a change point detection algorithm on the means (GLRT or CuSum)
    4. Return the detecting change times
    (Optional) 5. Compute the autocorrelation to verify that the corerelation is small.
    """
    start_time = 0
    end_time = 2000
    batch_size = 100
    my_service_rates = [10, 10]
    my_arrival_rates = [4, 11]
    time_of_changes = [0, 1000]
    simulation_type = "DEDS"

    #     1. Generate wait times for a M/M/1 queue for a time period [0,100] with a change of parameters at t=50
    #     from (\lambda=9, \mu=10) to (\lambda=4, \mu=10)
    if simulation_type == "DEDS":
        deds_sim = SingleServerMarkovianQueue(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                              time_of_changes)
        wait_times = deds_sim.simulate_deds_process(warm_up=30)
        recorded_times = deds_sim.get_recorded_times()
        glrt_threshold_t = 7
    elif simulation_type == "LADDER":
        ladder_point_creator = LadderPointQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                                my_service_rates,
                                                time_of_changes)
        wait_times = ladder_point_creator.simulate_ladder_point_process()
        recorded_times = ladder_point_creator.report_wait_record_times()
        glrt_threshold_t = 0.5

    print("Number of Wait Times: ", len(wait_times))
    print("Number of Recorded Times: ", len(recorded_times))
    assert (len(wait_times) == len(recorded_times))
    # 2. Get the means of 6-10 different batches and return the time stamp of the center of the batch
    batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, recorded_times,
                                                                             batch_size=batch_size)
    # 3. Run a change point detection algorithm on the m
    glrt_change_point_estimator = GLRTChangePointDetector(glrt_threshold_t, None, dist_type='gaussian')
    glrt_change_point_estimator.compute_change_point_locations(batch_mean_wait_times)
    deds_glrt_detection_statistics = DetectionStatistics(glrt_change_point_estimator.detection_result, batch_centers)
    glrt_detected_change_times = deds_glrt_detection_statistics.change_point_times
    print("GLRT with NBM detected changes at: ", glrt_detected_change_times,
          deds_glrt_detection_statistics.detection_values)
    glrt_stats = deds_glrt_detection_statistics.detection_values
    plot_wait_times_against_D_k_n(batch_mean_wait_times[:-1], batch_centers[:-1], glrt_detected_change_times,
                                  glrt_stats, np.quantile(glrt_stats, 0.75),
                                  time_of_changes[1:],
                                  "Figures/{sim_type}_SIM/change_point_detected_{sim_type}_batch_{size}.png".format(
                                      sim_type=simulation_type, size=batch_size))


#     plt.plot(batch_centers, batch_mean_wait_times, 'o')
#     plt.xlabel('t')
#     plt.ylabel('Wait times')
#     plt.title("NBM M/M/1 Wait Times")
#     plt.show()
#    plt.savefig('nbm_mm1_wait_times_deds.png', dpi=500)

def draw_random_wait_times_and_return_mean_variance(num_draws, lambda_param, mu_param):
    start_time = 0
    end_time = int(1.0 / lambda_param * num_draws) + 1
    my_service_rates = [mu_param]
    my_arrival_rates = [lambda_param]
    rho = lambda_param / float(mu_param)
    time_of_changes = [0]
    expected_wait_time_1 = lambda_param / float(mu_param)
    print("Expected Wait Time 1: ", expected_wait_time_1, end="\t")
    print("Variance Wait Time 1: ", expected_wait_time_1 + expected_wait_time_1 ** 2)

    expected_wait_time_2 = rho / (mu_param - lambda_param)
    var_wait_time_2 = rho * (2 - rho) / ((mu_param - lambda_param) * (mu_param - lambda_param))
    print("Expected Wait Time 2: ", expected_wait_time_2, end="\t")
    print("Variance Wait Time 2: ", var_wait_time_2)

    expected_wait_time_3 = (mu_param - lambda_param) / (mu_param * lambda_param)
    var_wait_time_3 = (1.0 / (mu_param ** 2) + 1.0 / (lambda_param ** 2))
    print("Expected Wait Time 3: ", expected_wait_time_3)
    print("Variance Wait Time 3: ", var_wait_time_3)
    ladder_point_creator = LadderPointQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                            my_service_rates,
                                            time_of_changes)
    wait_times = ladder_point_creator.simulate_ladder_point_process()
    mean_wait_time = np.mean(wait_times)
    var_wait_time = np.var(wait_times)
    print("Overall mean wait time at end of sim: {0:2.6f}".format(mean_wait_time), end="\t")
    print("Variance of wait time: {0:2.6f}.".format(var_wait_time))


def main():
    # run_expo_expo_test()
    # visualize_m_m_1_ladder_points()
    # visualize_m_m_1_deds()
    # run_m_m_1_test()
    # run_detection_tests_on_independent_wait_times()
    draw_random_wait_times_and_return_mean_variance(1000000, 80, 100)


if __name__ == "__main__":
    main()
