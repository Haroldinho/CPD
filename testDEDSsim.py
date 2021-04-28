"""
 Test Steady state results of the DEDS simulation

"""
import numpy as np
from generate_m_m_1_processes import SingleServerMarkovianQueue


def compute_mm1_mean(arrival_rate, service_rate):
    return (arrival_rate / service_rate) / (service_rate - arrival_rate)


def test_case_1():
    rho = 0.8
    mu = 1.0
    arrival_rate = rho * mu
    start_time = 0
    end_time = 20000000
    arrival_change_points = [0]
    service_change_points = [0]
    deds_creator = SingleServerMarkovianQueue(start_time, end_time, [arrival_rate],
                                              arrival_change_points, [mu], service_change_points)
    wait_times = deds_creator.simulate_deds_process()
    inter_arrival_times = deds_creator.return_inter_arrival_times()
    service_times = deds_creator.return_service_times()
    expected_wait_time = compute_mm1_mean(arrival_rate, mu)
    expected_inter_arr_time = 1.0 / arrival_rate
    expected_svce_time = 1.0 / mu
    print("Average wait time: {0:2.4f} against theoretical wait time of {1:2.4f}".format(np.mean(wait_times),
                                                                                         expected_wait_time))
    print("Average inter-arrival time: {0:2.4f} against theoretical inter-arrival time {1:2.4f}".format(
        np.mean(inter_arrival_times), expected_inter_arr_time))
    print("Average service time: {0:2.4f} against theoretical service time {1:2.4f}".format(np.mean(service_times),
                                                                                            expected_svce_time))


if __name__ == '__main__':
    test_case_1()
