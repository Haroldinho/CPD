import random
import time
import unittest

import numpy as np

from estimation_facility import compute_backward_updated_gaussian_statistic, compute_forward_updated_gaussian_statistic, \
    GaussianStatistic

"""
 Test the forward and backward formulas to compute variance
 as well as the compute_ross_updated gaussian statistic
"""


class TestGaussianRecursion(unittest.TestCase):
    def test_compute_forward_update(self):
        """
            Generate  a new updates
            For each new update
            Append the update to a vector
            Compute the variance of the new vector
            and compute the updated variance with the new element
        """
        i = 2
        MAX_LENGTH = 50000
        MAX_NUM = 5000
        rand_num_vec = [random.random() * MAX_NUM, random.random() * MAX_NUM]
        var_estimate = np.var(rand_num_vec, ddof=0)
        current_mean = np.mean(rand_num_vec)
        current_var = var_estimate
        updated_stat = GaussianStatistic(current_mean, current_var, i)
        while i < MAX_LENGTH:
            current_stat = updated_stat
            i += 1
            new_num = random.random() * MAX_NUM
            time_start_update = time.time()
            updated_stat = compute_forward_updated_gaussian_statistic(current_stat, new_num, i)
            time_to_update = time.time() - time_start_update
            new_var = updated_stat.variance
            rand_num_vec.append(new_num)
            time_start_var_comp = time.time()
            true_var = np.var(rand_num_vec, ddof=0)
            time_to_comp_var = time.time() - time_start_var_comp
            self.assertAlmostEqual(true_var, new_var, delta=1e-4)
            if (i % 10000) == 0:
                print("At iteration {0:d}, recursive_var={1:2.4f} and true var={2:2.4f}".format(i, new_var, true_var))
                print("\t with update time {0:2.6f} and computation time {1:2.6f}".format(time_to_update,
                                                                                          time_to_comp_var))

    def test_compute_backward_update(self):
        """
        Start with a very large number of elements and recomputing the variance as we remove elements
        """
        NUM_POINTS = 100000
        MAX_NUM = 5000
        # 1. Create a number of random variates uniformly distributed between 0 and MAX_NUM
        random_variates = []
        for i in range(NUM_POINTS):
            random_variates.append(random.random() * MAX_NUM)
        # 2. Compute the variance
        current_mean = np.mean(random_variates)
        current_var = np.var(random_variates, ddof=0)
        updated_stat = GaussianStatistic(current_mean, current_var, i)
        idx = 0
        while idx < NUM_POINTS - 2:
            current_stat = updated_stat
            time_start_update = time.time()
            updated_stat = compute_backward_updated_gaussian_statistic(current_stat, random_variates[idx])
            time_to_update = time.time() - time_start_update
            idx += 1

            time_start_var_comp = time.time()
            true_var = np.var(random_variates[idx:], ddof=0)
            time_to_comp_var = time.time() - time_start_var_comp
            new_var = updated_stat.variance
            self.assertAlmostEqual(true_var, new_var, delta=5e-1)
            if (i % 10000) == 0:
                print("At iteration {0:d}, recursive_var={1:2.4f} and true var={2:2.4f}".format(i, new_var, true_var))
                print("\t with update time {0:2.6f} and computation time {1:2.6f}".format(time_to_update,
                                                                                          time_to_comp_var))

        # 3. Iteratively remove an element
        #           Compute the variance using full variance computation and variance update
        # 4. Verify that the two are not
