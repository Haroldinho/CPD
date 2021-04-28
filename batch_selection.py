"""

  Change Point Detection part of thesis
  Selection of Batch size for the change point detection algorithm

  2020-2021
"""
import numpy as np
import pandas as pd


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


def find_nearest(array, value):
    array = np.asarray(array).reshape(-1, 1)
    idx = (np.abs(array - value)).argmin()
    # print("Shape of array: {}, shape of idx: {}".format(array.shape, idx.shape))
    # print(array)
    return_val = array[idx]
    if isinstance(return_val, (list, np.ndarray)):
        return_val = return_val[0]
    return return_val


class BatchSelectorOnAutocorrelation:
    """
        Class to select the batch size as a function of autocorrelation by leveraging Kemal's
        results
    """

    def __init__(self, desired_correlation):
        # Load the weights of the exponential function that returns the batch size
        self._a = None
        self._b = None
        self._c = None
        self._load_weights(desired_correlation)

    def _load_weights(self, desired_correlation):
        # load weight for nearest correlation
        coef_filename = "Data/KemalCorrelationDF.csv"
        weight_matrix = pd.read_csv(coef_filename)
        weight_matrix["correlation"] = pd.to_numeric(weight_matrix["correlation"])
        correlation_vector = np.array(weight_matrix["correlation"]).flatten()
        nearest_correlation = find_nearest(correlation_vector, desired_correlation)
        # print("Nearest correlation: {}".format(nearest_correlation))
        reduced_df = weight_matrix[correlation_vector == nearest_correlation]
        self._a = reduced_df["a"]
        self._b = reduced_df["b"]
        self._c = reduced_df["c"]

    def return_batch_size(self, rho):
        return int(exponential_func(rho, self._a, self._b, self._c)) + 1
