"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: utilities.py
Description: General ancillary functions to support other libraries in this package.
start date: April 8th 2020
Evaluation metrics: Detection delay, false alarm rates, ...
Author: Harold Nemo Adodo Nikoue
part of the partial observability thesis
"""

import time
from datetime import date
import json
import pickle as pkl


def my_timer(orig_func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print("{} ran in {} secs.".format(orig_func.__name__, t2))
        return result

    return wrapper


class PowerTestLogger:
    """
    In JSON, an object is an unordered set of name/value pairs
    Pickle can save any data type
    Here we have a dictionary of numpy array
    """

    def __init__(self, file_prefix, is_full_path=False, file_type='.json'):
        if is_full_path:
            self._file_name = file_prefix
        else:
            today = date.today()
            day_str = today.strftime("%m_%d_%Y")
            self._file_name = file_prefix + day_str + file_type

    def write_data(self, data_dict):
        # I overwrite every single time
        with open(self._file_name, 'wb') as outfile:
            pkl.dump(data_dict, outfile)
            # use indent to pretty print
            # json.dump(data_dict, outfile, indent=4)

    def load_data(self):
        with open(self._file_name, 'rb') as pickle_file:
            data = pkl.load(pickle_file)
        return data
