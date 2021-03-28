"""
Simple code to validate my understanding of the GLRT method
for Gaussian variables
as described by Ross 2014
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

from batch_means_methods import create_nonoverlapping_batch_means
from generate_m_m_1_processes import simulate_deds_return_wait_times
from utilities import PowerTestLogger


def compute_ross_gaussian_correction_factor(k, t):
    return t * (np.log(2. / t) + digamma((t - 1) / 2.)) - k * (np.log(2. / k) + digamma((k - 1) / 2.0)) - (t - k) * (
            np.log(2.0 / (t - k)) +
            digamma((t - k - 1) / 2.0))


def compute_var_statistic(x_vec):
    sample_mean = np.mean(x_vec)
    variance = 0
    for x in x_vec:
        variance += (x - sample_mean) * (x - sample_mean) / (len(x_vec) - 1)
    return variance


def compute_test_gaussian_statistic(x_vec):
    """
    Test to compute the Ross test statistic for Gaussian random variables
    :param x_vec: A vector of length n
    Iterate over different k  going from 2 to n-1
        Compute the variance up to k and after k
    """
    n = len(x_vec)
    if n < 5:
        return 0
    D_n = 0
    for k in range(2, n - 2):
        S_up_to_k = np.var(x_vec[:k])
        S_after_k = np.var(x_vec[k:])
        S_total = np.var(x_vec)
        correction_factor = compute_ross_gaussian_correction_factor(k, n)
        k = float(k)
        D_k_n = 2*(k * np.log(S_total / S_up_to_k) + (n - k) * np.log(S_total / S_after_k)) / correction_factor
        if D_k_n > D_n:
            D_n = D_k_n
    return D_n


def compute_ross_exponential_correction_factor(k, n):
    correction_factor = - 2 * (k * digamma(k) + (n - k) * digamma(n - k)
                               - n * digamma(n) + n * np.log(n) - k * np.log(k)
                               - (n - k) * np.log(n - k))
    return correction_factor


def compute_test_exponential_statistic(x_vec):
    """
    Test to compute the Ross test statistic for Exponential random variables
    :param x_vec: A vector of length n
    Iterate over different k  going from 2 to n-1
        Compute the variance up to k and after k
    """
    n = len(x_vec)
    if n < 5:
        return 0
    M_n = 0
    for k in range(2, n - 2):
        S_up_to_k = np.var(x_vec[:k])
        S_after_k = np.var(x_vec[k:])
        S_total = np.var(x_vec)
        correction_factor = compute_ross_exponential_correction_factor(k, n)
        k = float(k)
        M_k_n = -2 * (k * np.log(n / S_total) - (n - k) * np.log((n - k) / S_after_k)
                      - k * np.log(k / S_up_to_k)) / correction_factor
        if M_k_n > M_n:
            M_n = M_k_n
    return M_n


def run_gaussian_arl_estimation(h_t, num_iterations, dist_type="gaussian"):
    """
        2. While num_iter = 0 < N, the number of runs
            2.1 Simulate a random normal
            2.2. Update the Detection statistic
                2.3.1. If the detection statistic exceed h_t, records the number of random normals simulated in the loop
                as one ARL_0
                2.3.2. Else go back to 2.1
        3. Average the ARL_0
    """
    MAX_RUN_LENGTH = 1e5
    print("Testing GLRT with a threshold {} and {} iterations".format(h_t, num_iterations))
    iteration = 0
    arl_vec = [MAX_RUN_LENGTH for _ in range(num_iterations)]
    while iteration < num_iterations:
        if (iteration % 5) == 0:
            print("Iteration: ", iteration)
        x_vector = []
        run_length = 0
        while run_length < MAX_RUN_LENGTH:
            x_vector.append(np.random.normal())
            if dist_type == "gaussian":
                D_k = compute_test_gaussian_statistic(x_vector)
            else:
                D_k = compute_test_exponential_statistic(x_vector)
            # print("D_t= {0:2.4f} for t={1}".format(D_k, len(x_vector)))
            if D_k > h_t:
                # print("Break with D_k={0:2.4f}".format(D_k))
                arl_vec[iteration] = len(x_vector)
                break
            run_length += 1
        iteration += 1
    return np.nanmean(arl_vec)


def main_one(dist_type="gaussian"):
    """
    First obtain the Average Run Length before false detection (ARL_0) for different threhsholds
    for i.i.d. Normal(0,1).
    1. Pick a threshold h_t

    2. While num_iter = 0 < N, the number of runs
        2.1 Simulate a random normal
        2.2. Update the Detection statistic
        2.3.1. If the detection statistic exceed h_t, records the number of random normals simulated in the loop
                as one ARL_0
        2.3.2. Else go back to 2.1

    3. Average the ARL_0
    4. Go back to 1. with a new threshold
    5. Plot ARL_0 vs h_t
    """
    #     log_directory = "./Results/GLRT_ROSS/"
    #     log_name = log_directory + "glrt_ross_arl_0_results_{}".format(dist_type)
    #     arl_0_log = PowerTestLogger(log_name, is_full_path=False, file_type='pkl')
    num_iter = 600
    detection_thresholds = np.linspace(.1, 25, 30)
    arl_vector = []
    arl_dic = {}
    for threshold in detection_thresholds:
        arl_vector.append(run_gaussian_arl_estimation(threshold, num_iter, dist_type))
        arl_dic[arl_vector[-1]] = detection_thresholds
    #         arl_0_log.write_data(arl_dic)

    # plot Ross' nonlinear fit
    # equi_spaced_arl = np.linspace(np.min(arl_vector), int(np.max(arl_vector)))
    # equi_spaced_gamma = [1.0/arl for arl in equi_spaced_arl]
    # h_t_vec = [1.51 - 2.39 * np.log(gamma) + (3.65 + 0.76 * np.log(gamma))/(np.sqrt(t-7))]
    plt.plot(arl_vector, detection_thresholds, '.-')
    plt.xlabel('ARL_0')
    plt.ylabel('h_t')
    plt.title('GLRT for i.i.d N(0,1)')
    #     plt.savefig("Results/GLRT_ROSS/glrt_ross_validation_{}.png".format(dist_type), dpi=500)
    plt.show()


def obtain_batch_mean_wait_time(start_time, end_time, arr_rate, time_of_changes, srv_rate, batch_size):
    """
        Obtain Batch Mean Wait Times

    """
#     wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, arr_rate, time_of_changes,
#                                                               srv_rate)
    wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, arr_rate, time_of_changes,
                                                                srv_rate)
    batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts, batch_size)
    return batch_mean_wait_times


def run_wait_time_arl_estimation(h_t, num_iterations, lambda_param, mu_param, batch_size, end_time=200000):
    """
        Use Batch Means to compute ARL_0 by creating a sequence of wait times
        The false alarm probability is the ratio of the number of times a false change was detected
        divide the number of incorrect detections by the total number of detections
        There are three outcomes for a test:
        a) nothing gets detected
        b) a change gets detected before the actual change false positive
        c) a change gets detected after the actual change true positive
    """
    # if there is no detection, the time to false detection is at least as big as our testing window
    MAX_RUN_LENGTH = end_time
    print("Testing GLRT with a threshold {} and {} iterations".format(h_t, num_iterations))
    iteration = 0
    # for every iteration with a correct detection set the average run length to infinity or a very high value
    # and then for each detection put the correct value
    arl_vec = [MAX_RUN_LENGTH for _ in range(num_iterations)]
    start_time = 0
    my_arrival_rates = [lambda_param]
    my_service_rates = [mu_param]
    time_of_changes = [0]
    # NOTE: The simulation gets stuck for large end-times
    # TODO: Figure out why the simulation gets stuck and modify that code
    num_false_detections = 0
    while iteration < num_iterations:
        if (iteration % 5) == 0:
            print("Iteration: ", iteration)
        iteration += 1
        run_length = 0
        batch_centers = obtain_batch_mean_wait_time(start_time, end_time, my_arrival_rates, time_of_changes,
                                                    my_service_rates, batch_size)
        while run_length < len(batch_centers):
            # Perform GLRT to see if there is a change point in the current batch
            x_vector = batch_centers[:run_length + 1]
            D_k = compute_test_gaussian_statistic(x_vector)
            # print("D_t= {0:2.4f} for t={1}".format(D_k, len(x_vector)))
            if D_k > h_t:
                # print("Break with D_k={0:2.4f}".format(D_k))
                arl_vec[iteration] = len(x_vector) * batch_size
                num_false_detections += 1
                break
            run_length += 1

    false_alarm_prob = num_false_detections / num_iterations
    return np.median(arl_vec), false_alarm_prob


def reproduce_figure_1_b_ross():
    """
    Plot h_t vs t to reproduce Figure 1.b of Gordon Ross paper on Sequential Change Point Detection 2014
    :return: plot h_t vs. t
    """
    #  1. Simulate 2 millions sequences of independent N(0,1)
    #   Each sequence is naturally indexed as a vector
    #   Maybe put it in a numpy matrix
    #  2. Obtain the corresponding test statistic matrix
    #  3. Recursively compute gamma for each t and find the corresponding h_t
    #       Hence we will need a sort of binary search outer routine to help determine the right h_t by varying gamma
    #       and a function that compute gamma based on h_t and the number of elements to choose from for the recursion
    #   If t =1,
    #            Find the value of h_1 such that for each Test statistic D_1 for the entire vector,
    #               the number of test statistic over h_1 divided by the total number of sequences
    #               is as close to gamma as possible (that's why we need Bin. search)
    # For the next step (t=2), keep only the paths satisfying the threshold requirement
    #   If t=k,
    #           Find the value of h_k such that the number of test statistic D_k over the number of retained paths
    #           is greater than h_k with probability gamma
    #   4. Return a dictionary of h_t indexed by t


def main_wait_times(batch_size):
    """
        Use Batch mean wait times to set the ARL_0
    """
    log_directory = "./Results/GLRT_ROSS/"
    end_time = 350000
    num_iter = 100
    detection_thresholds = np.linspace(.1, 20, 15)
    arl_vector = []
    far_vector = []
    arl_dic = {}
    far_dic = {}
    mu_param = 10
    lambda_param = 8
    # batch_size = 75
    log_name = log_directory + "glrt_ross_arl_0_wait_time_lambda_{}_mu_{}_b_{}_".format(lambda_param, mu_param,
                                                                                        batch_size)
    far_log_name = log_name + "_far_"
    arl_0_log = PowerTestLogger(log_name, is_full_path=False, file_type='pkl')
    far_log = PowerTestLogger(far_log_name, is_full_path=False, file_type='pkl')
    for threshold in detection_thresholds:
        arl_0_score, far_score = run_wait_time_arl_estimation(threshold, num_iter, lambda_param, mu_param, batch_size,
                                                       end_time)
        arl_vector.append(arl_0_score)
        far_vector.append(far_score)
        arl_dic[arl_vector[-1]] = threshold
        far_dic[far_vector[-1]] = threshold
        arl_0_log.write_data(arl_dic)
        far_log.write_data(far_dic)

    # plot Ross' nonlinear fit
    # truncate far_vector
    x_detection_vec = []
    y_far_vec = []
    for i in range(len(far_vector)):
        if far_vector[i] < (end_time * 0.8):
            x_detection_vec.append(detection_thresholds[i])
            y_far_vec.append(arl_vector[i])
    plt.plot(x_detection_vec, y_far_vec, '.-')
    plt.ylabel('FAR')
    plt.xlabel('h_t')
    plt.title('GLRT for Batch={}'.format(batch_size))
    plt.savefig("Results/GLRT_ROSS/glrt_ross_wait_time_batch_{}_lambda_{}_mu_{}_far.png".format(batch_size,
                                                                                                lambda_param, mu_param),
                dpi=500)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    # main_one(dist_type="exponential")
    main_wait_times(1)
    main_wait_times(5)
    main_wait_times(10)
    main_wait_times(25)
    main_wait_times(50)
    main_wait_times(75)
    main_wait_times(100)
    main_wait_times(125)
    main_wait_times(150)
    main_wait_times(175)
    main_wait_times(200)


