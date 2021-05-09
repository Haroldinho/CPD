import unittest

import numpy as np
import pandas as pd


class TestCondProbabilityOfaChange(unittest.TestCase):
    """
    Test
    Make sure the sum of the probabilities add up to 1,
    """

    def setUp(self):
        main_file_name = "./Results/GLRT_ROSS/Performance_Tests/joint_conditional_probability.csv"
        joint_cond_prob_df = pd.read_csv(main_file_name)
        self.cond_prob_df = joint_cond_prob_df[["rho", "delta_rho", "Batch Size",
                                                "A+ | Change",
                                                "A- | Change",
                                                "W+ | Change",
                                                "W- | Change",
                                                "Q+ | Change",
                                                "Q- | Change",
                                                "A+ | No Change",
                                                "A- | No Change",
                                                "W+ | No Change",
                                                "W- | No Change",
                                                "Q+ | No Change",
                                                "Q- | No Change"
                                                ]]
        self._n = self.cond_prob_df.shape[0]
        self._vector_ones = np.ones(self._n)

    def test_age_add_up(self):
        self.assertIsNone(np.testing.assert_almost_equal(self.cond_prob_df["A+ | Change"] +
                                                         self.cond_prob_df["A- | Change"],
                                                         self._vector_ones))

    #         self.assertAlmostEqual(self.cond_prob_df["A+ | No Change"] + self.cond_prob_df["A- | No Change"],
    #                                self._vector_ones)

    def test_queue_add_up(self):
        self.assertAlmostEqual(self.cond_prob_df["Q+ | Change"] + self.cond_prob_df["Q- | Change"], 1)
        self.assertAlmostEqual(self.cond_prob_df["Q+ | No Change"] + self.cond_prob_df["Q- | No Change"], 1)

    def test_wait_add_up(self):
        self.assertAlmostEqual(self.cond_prob_df["W+ | Change"] + self.cond_prob_df["W- | Change"], 1)
        self.assertAlmostEqual(self.cond_prob_df["W+ | No Change"] + self.cond_prob_df["W- | No Change"], 1)


if __name__ == '__main__':
    unittest.main()
