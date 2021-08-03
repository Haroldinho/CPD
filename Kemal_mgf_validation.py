"""
  Chapter 2: Change Point Detection
  Validation of Kemal's results on autocorrelation of waiting times of Non-overlapping Batch Means for  M/M/1 queues

  2021-2022
"""
from typing import List

import numpy as np
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from tqdm import tqdm

from Kemal_autocorrelation_validation import compute_mm1_mean, compute_batch_waiting_times_variance
from generate_m_m_1_processes import simulate_deds_return_wait_times


@dataclass()
class SettingExperiment:
    constant_batch_size_list: List[int] = field(default_factory=list)
    constant_rho_list: List[float] = field(default_factory=list)

    def __init__(self):
        self.constant_batch_size_list = [20, 40, 80, 200, 500, 1000, 2000, 5000]
        self.constant_rho_list = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]


@dataclass()
class InitialParameter:
    list_of_t: List
    const_service_rate: float
    const_rate_diff: float
    sim_length: int
    const_num_iter: int = 1000
    batch_size: int = 200
    const_rho: float = 0.1
    const_arr_rate: float = 0.25
    warm_up_length: int = 2000

    def __init__(self):
        self.list_of_t = np.linspace(-0.15, 0.15, 20)
        self.const_service_rate = self.const_arr_rate / self.const_rho
        self.const_rate_diff = self.const_service_rate - self.const_arr_rate
        # simple heuristic to find required simulation window
        self.sim_length = get_desired_runtime(self.const_arr_rate, self.batch_size, self.warm_up_length)


def get_desired_runtime(arr_rate, batch_size, warm_up_length=2000):
    return int((warm_up_length + batch_size) / max(1, arr_rate))


def run_comparison_plots(a_general_setting: SettingExperiment, a_sim_param: InitialParameter):
    for idx_rho in tqdm(range(len(a_general_setting.constant_rho_list)), desc="Iterating over rho: "):
        rho = a_general_setting.constant_rho_list[idx_rho]
        plot_empirical_waiting_time_mgf_vs_gaussian_different_batch_sizes(
            a_sim_param,
            a_general_setting.constant_batch_size_list,
            rho
        )


def plot_empirical_waiting_time_mgf_vs_gaussian_different_batch_sizes(a_sim_param: InitialParameter, batch_sizes: List,
                                                                      rho: float):
    fig, ax = plt.subplots()
    t_abscissae = a_sim_param.list_of_t
    gaussian_mgf = [t * t / 2 for t in t_abscissae]
    ax.plot(t_abscissae, gaussian_mgf, label="Gaussian")
    for idx in tqdm(range(len(batch_sizes)), desc="Iterating over batch size: "):
        m = batch_sizes[idx]
        a_sim_param.batch_size = m
        list_of_log_mgf_at_t = get_estimates_of_log_normalized_mgf(a_sim_param)
        ax.plot(t_abscissae, list_of_log_mgf_at_t, lw=3, zorder=7, label=f"KDE for m={m}")
    ax.set_xlabel('t')
    ax.set_ylabel('log mgf')
    ax.set_title(f'rho = {rho}')
    ax.legend()
    plt.savefig('./Figures/MGF/log_mgf_vs_gaussian_rho_{}_.png'.format(int(rho * 100)))
    plt.close(fig)


def get_estimates_of_log_normalized_mgf(a_sim_param: InitialParameter):
    list_of_estimates_log_mgf = []
    list_of_estimates_mgf = []
    for index in tqdm(range(len(a_sim_param.list_of_t)), desc="Iterating over t: "):
        log_mgf, mgf = get_estimates_of_two_mgfs_for_one_t(
            a_sim_param.list_of_t[index],
            a_sim_param.const_num_iter,
            a_sim_param.const_arr_rate,
            a_sim_param.const_service_rate,
            a_sim_param.batch_size,
            a_sim_param.sim_length
        )
        list_of_estimates_log_mgf.append(log_mgf)
        list_of_estimates_mgf.append(mgf)
    return list_of_estimates_log_mgf


def get_estimates_of_two_mgfs_for_one_t(t: float, num_iter: int, arr_rate: float, serv_rate: float, batch_size: int,
                                        sim_length: int):
    """
    :param t: parameter of the mgf
    :param num_iter: number of Monte Carlo simulations
    :param arr_rate: arrival rate for all the iterations
    :param serv_rate: service rate for all the runs
    :param batch_size:
    :param sim_length: length of the simulation
    :return: the mgf of the waiting time
                the log of the mgf of the normalized waiting times
    """
    sums_of_waiting_times = get_sums_of_waiting_times_for_each_run(num_iter, arr_rate, serv_rate, batch_size,
                                                                   sim_length)
    waiting_time_mgf = compute_empirical_waiting_time_mgf(sums_of_waiting_times, t)
    # normalize the (sum of )waiting times
    normalized_waiting_time_mgfs = normalize_sums_of_waiting_times(sums_of_waiting_times, arr_rate, serv_rate,
                                                                   batch_size)
    log_mgf_waiting_times = compute_log_of_normalized_waiting_time_mgf(normalized_waiting_time_mgfs, t)
    return log_mgf_waiting_times, waiting_time_mgf


def normalize_sums_of_waiting_times(sums_of_waiting_times: List, arr_rate: float, serv_rate: float, batch_size: int):
    mean_waiting_times = [sum_w / batch_size for sum_w in sums_of_waiting_times]
    expected_waiting_time = compute_mm1_mean(arr_rate, serv_rate)
    variance_batch_waiting_times = compute_batch_waiting_times_variance(batch_size, arr_rate, serv_rate)
    normalized_batch_mean_waiting_times = [(mean_value - expected_waiting_time) / variance_batch_waiting_times
                                           for mean_value in mean_waiting_times]
    # will need to compute the expectation adn variance of the waiting times
    return normalized_batch_mean_waiting_times


def compute_empirical_waiting_time_mgf(sums_of_waiting_times: List, t: float):
    return np.mean(np.exp(np.array(sums_of_waiting_times) * t))


def compute_log_of_normalized_waiting_time_mgf(normalized_waiting_times_z_scores: List, t: float):
    sample_average_mgf = np.mean(np.exp(t * np.array(normalized_waiting_times_z_scores)))
    return np.log(sample_average_mgf)


def get_sums_of_waiting_times_for_each_run(num_iter: int, arr_rate: float, serv_rate: float, batch_size: int,
                                           sim_length: int):
    """
    :param num_iter: number of Monte Carlo simulations
    :param arr_rate: arrival rate for all the iterations
    :param serv_rate: service rate for all the runs
    :param batch_size:
    :param sim_length: length of the simulation
    :return: a list of the sums of the waiting times for the given batch size
    """
    list_of_sums = []
    for _ in range(num_iter):
        list_of_sums.append(get_one_sum_of_waiting_time_for_one_sim(arr_rate, serv_rate, batch_size, sim_length))
    return list_of_sums


def get_one_sum_of_waiting_time_for_one_sim(arr_rate: float, serv_rate: float, batch_size: int,
                                            sim_length: int):
    """
    :param arr_rate: arrival rate for all the iterations
    :param serv_rate: service rate for all the runs
    :param batch_size:
    :param sim_length: length of the simulation
    :return: one sum of the waiting times for the given batch size for one run
    """
    start_time = 0
    end_time = sim_length
    my_arrival_rates = [arr_rate, arr_rate]
    time_of_changes = [end_time]
    my_service_rates = [serv_rate, serv_rate]
    wait_times, _ = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates, time_of_changes,
                                                    my_service_rates)
    retained_wait_times = wait_times[-batch_size:]
    return sum(retained_wait_times)


if __name__ == "__main__":
    my_general_params = SettingExperiment()
    my_initial_params = InitialParameter()
    run_comparison_plots(my_general_params, my_initial_params)
