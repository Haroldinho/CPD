import unittest
from generate_stochastic_streams import generate_change_points_geom, generate_change_points_uniform
from performance_statistics_facility import AverageRunLength
from numpy.random import uniform
import numpy as np
from generate_m_m_1_processes import LadderPointQueue, SingleServerMarkovianQueue
from generate_m_m_1_processes import simulate_deds_return_wait_times, simulate_ladder_point_process
import random
from Kemal_autocorrelation_validation import compute_nobm_lag_autocorrelation
from scipy.stats import pearsonr
from batch_means_methods import create_nonoverlapping_batch_means


class TestComputingPerfStats(unittest.TestCase):
    def setUp(self):
        self.true_changes = uniform(low=0, high=100, size=10)
        self.detected_changed_points = uniform(low=0, high=100, size=20)
        self._arl = AverageRunLength(self.true_changes, self.detected_changed_points)

    def tearDown(self) -> None:
        pass

    def test_correct_length_marked_changes(self):
        """
        Verify that the number of marked change points correspond to the number of detected changes
        :return:
        """
        marked_change_points = self._arl.marked_changed_points
        self.assertEqual(len(self.detected_changed_points), len(marked_change_points))

    def test_detection_delays_are_all_positive_numbers(self):
        """

        :return:
        """
        self._arl.compute_arl_1()
        self.assertGreaterEqual(self._arl.detection_delays, [0] * len(self._arl.detection_delays))


class TestStreamGenerations(unittest.TestCase):
    def test_change_point_limits_geom(self):
        my_list_rates = uniform(4, 20, 3)
        interval_length = 10
        change_point_list = generate_change_points_geom(my_list_rates, interval_length)
        self.assertLessEqual(max(change_point_list), interval_length, msg="Change Points fall outside bounds")

    def test_change_point_limits_uniform(self):
        my_list_rates = uniform(4, 20, 3)
        interval_length = 10
        change_point_list = generate_change_points_uniform(my_list_rates, interval_length)
        self.assertLessEqual(max(change_point_list), interval_length, msg="Change Points fall outside bounds")


class TestWaitTimesGenerationLadder(unittest.TestCase):
    """
    Test
    a) the average sample wait times should be within 10% for a stream of <500 wait times.
    """

    def setUp(self):
        self._num_points = 500
        self._start_time = 0
        self._end_time = 15000
        self._arrival_change_points = [0]
        self._service_change_points = [0]
        self._service_rate = 10.0
        self.rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.rho = 0.9
        self._arrival_rate = self.rho * self._service_rate
        print("Service rate: ", self._service_rate)
        print("Arrival rate: ", self._arrival_rate)
        print("rho: {}\n".format(self._arrival_rate / self._service_rate))

    def test_ladder_queue_mean_waiting_time(self):
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                print("Service rate: ", self._service_rate)
                print("Arrival rate: ", arrival_rate)
                print("rho: {}\n".format(arrival_rate / self._service_rate))
                self._expected_wait_time = rho / (self._service_rate - arrival_rate)
                wait_times, wait_times_ts = simulate_ladder_point_process(self._start_time, self._end_time,
                                                                          [arrival_rate],
                                                                          self._arrival_change_points,
                                                                          [self._service_rate])
                count = max(10, int(len(wait_times) / 50))
                self.assertAlmostEqual(np.mean(wait_times[count:]), self._expected_wait_time,
                                       delta=0.1 * self._expected_wait_time)

    def test_ladder_queue_var_waiting_time(self):
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                print("Service rate: ", self._service_rate)
                print("Arrival rate: ", arrival_rate)
                print("rho: {}\n".format(arrival_rate / self._service_rate))
                self._variance_wait_time = rho * (2 - rho) / ((self._service_rate - arrival_rate) ** 2)
                wait_times, wait_times_ts = simulate_ladder_point_process(self._start_time, self._end_time,
                                                                          [arrival_rate],
                                                                          self._arrival_change_points,
                                                                          [self._service_rate])
                count = max(10, int(len(wait_times) / 50))
                self.assertAlmostEqual(np.var(wait_times[count:]), self._variance_wait_time,
                                       delta=0.1 * self._variance_wait_time)

    def test_ladder_point_queue_consistency(self):
        ladder_point_creator = LadderPointQueue(self._start_time, self._end_time, [self._arrival_rate],
                                                self._arrival_change_points, [self._service_rate],
                                                self._service_change_points)
        wait_times = ladder_point_creator.simulate_ladder_point_process()
        idle_times = ladder_point_creator.get_idle_times()
        ladder_times = ladder_point_creator.get_ladder_times()
        self.assertEqual(len(wait_times) + len(idle_times), len(ladder_times))

    def test_ladder_autocorrelation_batch_1(self):
        batch_size = 1
        lag = 1
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                wait_times, wait_times_ts = simulate_ladder_point_process(self._start_time, self._end_time,
                                                                          [arrival_rate], self._arrival_change_points,
                                                                          [self._service_rate])
                nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                   batch_size)
                kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, arrival_rate / self._service_rate)
                count = max(10, int(len(wait_times) / 10))
                test_array = nobm_wait_times[count:]
                auto_corr, p_value = pearsonr(test_array[:-lag], test_array[lag:])
                self.assertAlmostEqual(auto_corr, kemal_autocorr, delta=0.05 * kemal_autocorr)

    def test_ladder_autocorrelation_batch_5(self):
        batch_size = 5
        lag = 1
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                wait_times, wait_times_ts = simulate_ladder_point_process(self._start_time, self._end_time,
                                                                          [arrival_rate], self._arrival_change_points,
                                                                          [self._service_rate])
                nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                   batch_size)
                kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, arrival_rate / self._service_rate)
                count = max(10, int(len(wait_times) / 10))
                test_array = nobm_wait_times[count:]
                auto_corr, p_value = pearsonr(test_array[:-lag], test_array[lag:])
                self.assertAlmostEqual(auto_corr, kemal_autocorr, delta=0.05 * kemal_autocorr)

    def test_ladder_autocorrelation_batch_25(self):
        batch_size = 25
        lag = 1
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                wait_times, wait_times_ts = simulate_ladder_point_process(self._start_time, self._end_time,
                                                                          [arrival_rate], self._arrival_change_points,
                                                                          [self._service_rate])
                nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                   batch_size)
                kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, arrival_rate / self._service_rate)
                count = max(10, int(len(wait_times) / 10))
                test_array = nobm_wait_times[count:]
                auto_corr, p_value = pearsonr(test_array[:-lag], test_array[lag:])
                self.assertAlmostEqual(auto_corr, kemal_autocorr, delta=0.05 * kemal_autocorr)


class TestWaitTimesGenerationDEDS(unittest.TestCase):
    """
    Test
    a) the average sample wait times should be within 10% for a stream of <500 wait times.
    """

    def setUp(self):
        self._num_points = 500
        self._start_time = 0
        self._end_time = 15000
        self._arrival_change_points = [0]
        self._service_change_points = [0]
        #        self._service_rate = random.uniform(1.0, 100)
        self._service_rate = 1.0
        #         rho = random.uniform(0, 1.0)
        self.rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.rho = 0.9
        self._arrival_rate = self.rho * self._service_rate
        print("Service rate: ", self._service_rate)
        print("Arrival rate: ", self._arrival_rate)
        print("rho: {}\n".format(self._arrival_rate / self._service_rate))

    def simulate_queue_produces_positive_wait_times(self):
        deds_creator = SingleServerMarkovianQueue(self._start_time, self._end_time, [self._arrival_rate],
                                                  self._arrival_change_points, [self._service_rate],
                                                  self._service_change_points)
        wait_times = deds_creator.simulate_deds_process()
        self.assertGreaterEqual(wait_times, 0)

    def test_deds_queue_mean_waiting_time(self):
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                print("Service rate: ", self._service_rate)
                print("Arrival rate: ", arrival_rate)
                print("rho: {}\n".format(arrival_rate / self._service_rate))
                self._expected_wait_time = rho / (self._service_rate - arrival_rate)
                wait_times, wait_times_ts = simulate_deds_return_wait_times(self._start_time, self._end_time,
                                                                            [arrival_rate],
                                                                            self._arrival_change_points,
                                                                            [self._service_rate])
                count = max(10, int(len(wait_times) / 50))
                self.assertAlmostEqual(np.mean(wait_times[count:]), self._expected_wait_time,
                                       delta=0.1 * self._expected_wait_time)

    def test_deds_queue_var_waiting_time(self):
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                print("Service rate: ", self._service_rate)
                print("Arrival rate: ", arrival_rate)
                print("rho: {}\n".format(arrival_rate / self._service_rate))
                self._variance_wait_time = rho * (2 - rho) / ((self._service_rate - arrival_rate) ** 2)
                wait_times, wait_times_ts = simulate_deds_return_wait_times(self._start_time, self._end_time,
                                                                            [arrival_rate],
                                                                            self._arrival_change_points,
                                                                            [self._service_rate])
                count = max(10, int(len(wait_times) / 50))
                self.assertAlmostEqual(np.var(wait_times[count:]), self._variance_wait_time,
                                       delta=0.1 * self._variance_wait_time)

    def test_deds_queue_cycle_times(self):
        deds_creator = SingleServerMarkovianQueue(self._start_time, self._end_time, [self._arrival_rate],
                                                  self._arrival_change_points, [self._service_rate],
                                                  self._service_change_points)
        wait_times = deds_creator.simulate_deds_process()
        cycle_times = deds_creator.get_cycle_times()
        self._expected_wait_time = (self._arrival_rate / self._service_rate) / (self._service_rate - self._arrival_rate)
        count = max(10, int(len(wait_times) / 50))
        self.assertAlmostEqual(np.mean(cycle_times[count:]), self._expected_wait_time + 1.0 / self._service_rate,
                               delta=0.1)

    def test_deds_queue_num_arrivals(self):
        deds_creator = SingleServerMarkovianQueue(self._start_time, self._end_time, [self._arrival_rate],
                                                  self._arrival_change_points, [self._service_rate],
                                                  self._service_change_points)
        deds_creator.generate_arrivals()
        num_arrivals = len(deds_creator.return_arrival_times())
        expected_num_arrivals = self._arrival_rate * (self._end_time - self._start_time)
        self.assertAlmostEqual(num_arrivals, expected_num_arrivals, delta=0.05 * expected_num_arrivals)

    def test_deds_autocorrelation_batch_1(self):
        batch_size = 1
        lag = 1
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                wait_times, wait_times_ts = simulate_deds_return_wait_times(self._start_time, self._end_time,
                                                                            [arrival_rate],
                                                                            self._arrival_change_points,
                                                                            [self._service_rate])
                nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                   batch_size)
                kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, arrival_rate / self._service_rate)
                count = max(10, int(len(wait_times) / 10))
                test_array = nobm_wait_times[count:]
                auto_corr, p_value = pearsonr(test_array[:-lag], test_array[lag:])
                self.assertAlmostEqual(auto_corr, kemal_autocorr, delta=0.05 * kemal_autocorr)

    def test_deds_autocorrelation_batch_5(self):
        batch_size = 5
        lag = 1
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                wait_times, wait_times_ts = simulate_deds_return_wait_times(self._start_time, self._end_time,
                                                                            [arrival_rate],
                                                                            self._arrival_change_points,
                                                                            [self._service_rate])
                nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                   batch_size)
                kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, arrival_rate / self._service_rate)
                count = max(10, int(len(wait_times) / 10))
                test_array = nobm_wait_times[count:]
                auto_corr, p_value = pearsonr(test_array[:-lag], test_array[lag:])
                self.assertAlmostEqual(auto_corr, kemal_autocorr, delta=0.05 * kemal_autocorr)

    def test_deds_autocorrelation_batch_25(self):
        batch_size = 25
        lag = 1
        for rho in self.rho_list:
            with self.subTest(rho=rho):
                arrival_rate = rho * self._service_rate
                wait_times, wait_times_ts = simulate_deds_return_wait_times(self._start_time, self._end_time,
                                                                            [arrival_rate],
                                                                            self._arrival_change_points,
                                                                            [self._service_rate])
                nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                   batch_size)
                kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, arrival_rate / self._service_rate)
                count = max(10, int(len(wait_times) / 10))
                test_array = nobm_wait_times[count:]
                auto_corr, p_value = pearsonr(test_array[:-lag], test_array[lag:])
                self.assertAlmostEqual(auto_corr, kemal_autocorr, delta=0.05 * kemal_autocorr)


if __name__ == '__main__':
    unittest.main()
