"""
                    Change Point Detection
        Implement Batch Means Methods to reduce the correlation of the in-sample estimates
"""
from typing import List, Tuple

import numpy as np


# @my_timer
def create_nonoverlapping_batch_means(x_data, timestamps, batch_size=5) -> Tuple[List[float], List[float]]:
    """
    :param x_data: vector/array of input data of size n
    :param timestamps: timestamp of each point
    :param batch_size: the desired size of the batches
    :return: vector/array of the mean of each batch, vector/array of the centroid (median) time of each batch
    """
    # print("Running batch means: [{},{}] on {} datapoints".format(timestamps[0], timestamps[-1], len(x_data)))
    assert (timestamps[-1] == timestamps[len(x_data) - 1])
    num_batches = int(np.ceil(len(x_data) / np.float(batch_size)))
    reduced_data = [0 for _ in range(num_batches)]
    recorded_times = [0 for _ in range(num_batches)]
    assert (num_batches * batch_size >= len(x_data))
    for i in range(num_batches):
        start_batch_idx = i * batch_size
        end_batch_idx = min(len(x_data) - 1, (i + 1) * batch_size)
        # print("Batch period idx: [{}, {}]".format(start_batch_idx, end_batch_idx), end='\t\t')
        # print("Batch time: [{}, {}]".format(timestamps[start_batch_idx], timestamps[end_batch_idx]))
        if start_batch_idx == len(x_data) - 1:
            continue
        window_data = x_data[start_batch_idx:end_batch_idx]
        # the mean of the batch
        reduced_data[i] = np.mean(window_data)
        recorded_times[i] = np.median(timestamps[start_batch_idx:end_batch_idx])
        #bigger values will influence the timestamp more
    return reduced_data, recorded_times


def sample_and_skip(x_data, stride=1):
    reduced_data = []
    for i in range(0, len(x_data), stride):
        reduced_data.append(x_data[i])
    return reduced_data
