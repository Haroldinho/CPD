"""
  Chapter 2: Change Point Detection
  Validation of Kemal's results on autocorrelation of waiting times of Non-overlapping Batch Means for  M/M/1 queues

  2021-2022
"""
from typing import List

import pickle as pkl
import numpy as np
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from tqdm import tqdm

from Kemal_autocorrelation_validation import compute_mm1_mean, compute_batch_waiting_times_variance
from batch_means_methods import create_nonoverlapping_batch_means
from generate_m_m_1_processes import simulate_deds_return_wait_times


@dataclass()
class SettingExperiment:
    constant_batch_size_list: List[int] = field(default_factory=list)
    constant_rho_list: List[float] = field(default_factory=list)

    def __init__(self):
        self.constant_batch_size_list = [20, 80, 200, 500, 2000]
        self.constant_rho_list = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]


@dataclass()
class InitialParameter:
    list_of_t: List
    const_service_rate: float
    const_rate_diff: float
    sim_length: int
    const_num_iter: int = 8000
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


def plot_quadratic_fit(ax, x_points, y_points, line_style, label):
    z = np.polyfit(x_points, y_points, 2)
    p = np.poly1d(z)
    y_quad = p(np.linspace(min(x_points), max(x_points)))
    ax.plot(x_points, y_points, '.', x_points, y_quad, line_style, label=label)


def plot_empirical_waiting_time_mgf_vs_gaussian_different_batch_sizes(a_sim_param: InitialParameter, batch_sizes: List,
                                                                      rho: float):
    fig, ax = plt.subplots()
    t_abscissae = a_sim_param.list_of_t
    gaussian_mgf = [t * t / 2 for t in t_abscissae]
    ax.plot(t_abscissae, gaussian_mgf, label="Gaussian")
    file_path_prefix = "./Data/MGF/"
    log_mgf_results_by_batch_size = []
    lines_styles = ["-", "--", "-.", ":", 'o']
    for idx in tqdm(range(len(batch_sizes)), desc="Iterating over batch size: "):
        m = batch_sizes[idx]
        a_sim_param.batch_size = m
        list_of_log_mgf_at_t = get_estimates_of_log_normalized_mgf(a_sim_param)
        log_mgf_results_by_batch_size[m] = list_of_log_mgf_at_t
        plot_quadratic_fit(ax, t_abscissae, list_of_log_mgf_at_t, lines_styles[idx], f"KDE for m={m}")
    #        ax.plot(t_abscissae, list_of_log_mgf_at_t, lw=3, zorder=7, label=f"KDE for m={m}")
    data_to_save_dict = {"list_of_t": t_abscissae, "log_mgfs": log_mgf_results_by_batch_size}
    pkl.dump(data_to_save_dict, file_path_prefix + f"_rho_{int(rho * 100)}.pkl")
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
        log_mgf = get_estimates_of_log_mgf_for_one_t(
            a_sim_param.list_of_t[index],
            a_sim_param.const_num_iter,
            a_sim_param.const_arr_rate,
            a_sim_param.const_service_rate,
            a_sim_param.batch_size,
            a_sim_param.sim_length
        )
        list_of_estimates_log_mgf.append(log_mgf)
    return list_of_estimates_log_mgf


def get_estimates_of_log_mgf_for_one_t(t: float, num_iter: int, arr_rate: float, serv_rate: float, batch_size: int,
                                       sim_length: int):
    """
    :param t: parameter of the mgf
    :param num_iter: number of Monte Carlo simulations
    :param arr_rate: arrival rate for all the iterations
    :param serv_rate: service rate for all the runs
    :param batch_size:
    :param sim_length: length of the simulation
    :return:
                the log of the mgf of the normalized waiting times
    """
    means_of_waiting_times = get_means_of_waiting_times_for_each_run(num_iter, arr_rate, serv_rate,
                                                                     batch_size, sim_length)
    normalized_waiting_time_mgfs = normalize_means_of_waiting_times(means_of_waiting_times, arr_rate, serv_rate,
                                                                    batch_size)
    log_mgf_waiting_times = compute_log_of_normalized_waiting_time_mgf(normalized_waiting_time_mgfs, t)
    return log_mgf_waiting_times


def normalize_means_of_waiting_times(means_waiting_times: List, arr_rate: float, serv_rate: float, batch_size: int):
    expected_waiting_time = compute_mm1_mean(arr_rate, serv_rate)
    variance_batch_waiting_times = compute_batch_waiting_times_variance(batch_size, arr_rate, serv_rate)
    normalized_batch_mean_waiting_times = [(mean_value - expected_waiting_time) / np.sqrt(variance_batch_waiting_times)
                                           for mean_value in means_waiting_times]
    # will need to compute the expectation adn variance of the waiting times
    return normalized_batch_mean_waiting_times


def compute_empirical_waiting_time_mgf(sums_of_waiting_times: List, t: float):
    return np.mean(np.exp(np.array(sums_of_waiting_times) * t))


def compute_log_of_normalized_waiting_time_mgf(normalized_waiting_times_z_scores: List, t: float):
    mgfs = np.exp(t * np.array(normalized_waiting_times_z_scores))
    sample_average_mgf = np.mean(mgfs)
    print(f"The coefficient of variation is: {sample_average_mgf / np.std(mgfs)}")
    return np.log(sample_average_mgf)


def get_means_of_waiting_times_for_each_run(num_iter: int, arr_rate: float, serv_rate: float,
                                            batch_size: int, sim_length: int, warm_up_period: int = 0):
    """
    :param num_iter: number of Monte Carlo simulations
    :param arr_rate: arrival rate for all the iterations
    :param serv_rate: service rate for all the runs
    :param batch_size:
    :param sim_length: length of the simulation
    :param warm_up_period:
    :return: a list of the means of the waiting times for the given batch size
    """
    list_of_means = []
    for _ in range(num_iter):
        list_of_means.append(get_waiting_time_batch_means_for_one_sim(arr_rate, serv_rate, batch_size, sim_length,
                                                                      warm_up_period))
    return list_of_means


def get_waiting_time_batch_means_for_one_sim(arr_rate: float, serv_rate: float, batch_size: int, sim_length: int,
                                             warm_up_period: int = 0):
    """
    :param arr_rate: arrival rate for all the iterations
    :param serv_rate: service rate for all the runs
    :param batch_size:
    :param sim_length: length of the simulation
    :param warm_up_period: part of the data to disgard
    :return: waiting_time non-overlapping batch means
    """
    start_time = 0
    end_time = sim_length
    my_arrival_rates = [arr_rate, arr_rate]
    time_of_changes = [end_time]
    my_service_rates = [serv_rate, serv_rate]
    wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates, time_of_changes,
                                                                my_service_rates)
    wait_times_ts = wait_times_ts[max(warm_up_period, 0):]
    wait_times = wait_times[max(warm_up_period, 0):]
    nobm_wait_times, _ = create_nonoverlapping_batch_means(wait_times, wait_times_ts, batch_size)
    return np.mean(nobm_wait_times)


if __name__ == "__main__":
    my_general_params = SettingExperiment()
    my_initial_params = InitialParameter()
    run_comparison_plots(my_general_params, my_initial_params)
