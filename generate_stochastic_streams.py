"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: generate_stochastic_streams.py
Description: Create sequences of random variates for a given stochastic processes with or without independence
Author: Harold Nemo Adodo Nikoue
part of the partial observability thesis
"""

from typing import List

from numpy.random import uniform
from scipy.stats import expon, geom


def generate_change_points_uniform(rates: List[float], interval_length: int) -> List[float]:
    # Generate randomly distributed change point times according to a probability p_k
    num_rates = len(rates)
    change_points = sorted(uniform(0, interval_length, num_rates))
    return change_points


def generate_change_points_geom(rates: List[float], interval_length: int) -> List[float]:
    # Generate randomly distributed change point times according to a probability p_k
    num_rates = len(rates)
    proportion = 1.0 / num_rates
    change_points = []
    start_time = 0
    for rate in rates:
        change_points.append(min(start_time, interval_length))
        start_time += geom.rvs(proportion)
    return change_points


def generate_non_homogeneous_expo_variables(rates: List[float], length: int, list_of_change_times=None):
    """
    Generate a stream of exponential random times for a non-homogeneous Poisson process
    for which the distribution change after each change point to a new rate
    :param list_of_change_times: (Optional) list of times at which the rates change
    :param rates: define the rate of the non-homogeneous Poisson process
    :param length: Total length of the time interval for the experiment
    :return: Arrival times (cumulative sums of the exponential times
    """
    if not list_of_change_times:
        list_of_change_times = generate_change_points_geom(rates, length)
    current_time = 0
    current_change_time_idx = 0
    arrival_times = []
    actual_change_point_times = []
    while current_time < length:
        arrival_times.append(current_time)
        if (current_change_time_idx + 1) < len(list_of_change_times) and current_time > list_of_change_times[
            current_change_time_idx + 1]:
            current_change_time_idx = current_change_time_idx + 1
            actual_change_point_times.append(arrival_times[-2])
        inter_arrival_duration = expon.rvs(1.0 / rates[current_change_time_idx])
        current_time += inter_arrival_duration
    return arrival_times, actual_change_point_times
