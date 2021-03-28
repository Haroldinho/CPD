"""
Test and validate the interpolation results for
"""

from operator import itemgetter

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicSpline

from utilities import PowerTestLogger


def is_sorted(l):
    return all(l[i] < l[i + 1] for i in range(len(l) - 1))


def create_list_of_tuples(list_a, list_b):
    assert (len(list_a) == len(list_b))
    return [(list_a[i], list_b[i]) for i in range(len(list_a))]


def load_dataframe(pkl_directory, pkl_files, batch_sizes):
    index = 0
    long_form_dataframe = pd.DataFrame(columns=["Batch Size", "Threshold", "ARL_0"])
    for idx, file_name in enumerate(pkl_files):
        pkl_logger = PowerTestLogger(pkl_directory + file_name, is_full_path=True)
        batch = batch_sizes[idx]
        pkl_data = pkl_logger.load_data()
        detection_thresholds_lists = list(pkl_data.values())
        arl_0_list = list(pkl_data.keys())
        # I am creating a dictionary
        if isinstance(detection_thresholds_lists[0], (list, tuple, np.ndarray)):
            detection_thresholds = detection_thresholds_lists[0]
            for idx in range(len(arl_0_list)):
                threshold = detection_thresholds[idx]
                values_to_add = [batch, threshold, arl_0_list[idx]]
                long_form_dataframe.loc[index] = values_to_add
                index += 1
        else:
            for arl_0, threshold in pkl_data.items():
                values_to_add = [batch, threshold, arl_0]
                long_form_dataframe.loc[index] = values_to_add
                index += 1
    return long_form_dataframe


def return_ross_threshold(gamma, t):
    return 1.51 - 2.39 * np.log(gamma) + (3.65 + 0.76 * np.log(gamma)) / np.sqrt(t - 7)


class ThresholdSelector:
    """
    Be careful, the solution will not make sense outside of the solution bound.
    right now for ARL_0 between min(ARL_0) and 150,000
    """
    def __init__(self, folder):
        self._folder_root = folder
        self._batch_sizes = {}
        # Create a long form dataframe from the data
        self._arl_threshold_batch_df = self.load_threshold_data()
        self._interp = self.create_grid_interpolant()
        self._cs = self.create_cubic_spline_func()

    def load_threshold_data(self):
        """
        Load different files

        """
        self._batch_sizes = [1, 5, 10, 25, 50, 75, 80, 100, 125, 150, 175, 200]
        file_names = [
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_1_10_25_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_5_10_31_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_10_11_04_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_25_11_07_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_50_11_10_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_75_11_13_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_80_05_23_2020.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_100_11_15_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_125_11_18_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_150_11_21_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_175_11_23_20.pkl",
            "glrt_ross_arl_0_wait_time_lambda_8_mu_10_b_200_10_11_20.pkl"
        ]
        return load_dataframe(self._folder_root, file_names, self._batch_sizes)

    def create_grid_interpolant(self):
        """
        Interpolate over batch size and arl_0
        """

        points = list(zip(self._arl_threshold_batch_df["Batch Size"], self._arl_threshold_batch_df["ARL_0"]))
        values = self._arl_threshold_batch_df["Threshold"]
#         for _, row in self._arl_threshold_batch_df:
#             points.append((row["Batch Size"], row["ARL_0"]))
#             values.append("Threshold")
        points = np.array(points)
        values = np.array(values)
        print("Shape of points ({})".format(points.shape))
        print("Shape of values {}".format(values.shape))
        interp = scipy.interpolate.LinearNDInterpolator(points, values, fill_value=0)
        return interp

    def create_cubic_spline_func(self):
        cubic_spline_dic = {}
        for batch_size in self._batch_sizes:
            reduced_df = self._arl_threshold_batch_df[self._arl_threshold_batch_df["Batch Size"] == batch_size]
            reduced_df = reduced_df[reduced_df["ARL_0"] <= 200000]
            reduced_df.sort_values(by=["Threshold"], inplace=True, ascending=True)
            y_thresholds = reduced_df["Threshold"].tolist()
            x_arl = reduced_df["ARL_0"].tolist()
            if not is_sorted(x_arl):
                # sort x_arl and y_threshold jointly
                # create a list of tuples
                list_tuple = create_list_of_tuples(x_arl, y_thresholds)
                sorted_list_tuples = sorted(list_tuple, key=itemgetter(0))
                x_arl, y_arl = map(list, zip(*sorted_list_tuples))
            cubic_spline_dic[batch_size] = CubicSpline(x_arl, y_thresholds, bc_type="not-a-knot", extrapolate=True)
        #             cubic_spline_dic[batch_size] = InterpolatedUnivariateSpline(x_arl, y_thresholds, ext=3)
        return cubic_spline_dic

    def get_threshold(self, batch_size, arl_0):
        """
        Get the threshold that corresponds to the desired ARL_0 for a given batch size
        """
        # reduce the dataframe to the desired dataframe based on batch size

        #         return self._cs[batch_size](arl_0)
        return self._interp(batch_size, arl_0)


if __name__ == "__main__":
    pkl_directory = "./Results/GLRT_ROSS/ARL_0/"
    h_selector = ThresholdSelector(pkl_directory)
    interesting_pairs = [(200, 15000), (200, 50000),
                         (200, 100000), (200, 120000), (200, 125000), (200, 140000), (200, 150000),
                         (200, 175000)]
    for batch, desired_arl in interesting_pairs:
        h_t = h_selector.get_threshold(batch, desired_arl)
        print("For a desired ARL_0={0:2.4f} and batch size={1:d}".format(desired_arl, batch), end=", ")
        print("the threshold is {0:3.4f}.".format(h_t))





