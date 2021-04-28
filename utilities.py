"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: utilities.py
Description: General ancillary functions to support other libraries in this package.
start date: April 8th 2020
Evaluation metrics: Detection delay, false alarm rates, ...
Author: Harold Nemo Adodo Nikoue
part of the partial observability thesis
"""

import pickle as pkl
import time
from datetime import date
import pandas as pd


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

    def __init__(self, file_prefix, is_full_path=False, file_type='json', dataframe=None, is_dataframe=False):
        """
        :param file_prefix: Beginning of file
        :param is_full_path: Is it the full path of the file or do we need to construct it
        :param file_type: if we need to construct is it a .json or .pkl file
        """
        self._df = dataframe
        if is_full_path:
            self._file_name = file_prefix
            self._is_dataframe = is_dataframe
        else:
            self._is_dataframe = isinstance(dataframe, pd.DataFrame)
            today = date.today()
            day_str = today.strftime("%m_%d_%y")
            self._file_name = file_prefix + day_str + "." + file_type

    def return_dataframe(self):
        return self._df

    def write_data(self, data):
        """
        If we are dealing with a dataframe, data should be a dataframe
        otherwise a dictionary
        """
        if self._is_dataframe:
            data_df = data
            data_df.to_pickle(self._file_name)
        else:
            data_dict = data
            # I overwrite every single time
            with open(self._file_name, 'wb') as outfile:
                pkl.dump(data_dict, outfile)
                # use indent to pretty print
                # json.dump(data_dict, outfile, indent=4)

    def load_dataframe(self):
        return pd.read_pickle(self._file_name)

    def load_data(self):
        if self._is_dataframe:
            with open(self._file_name, 'rb') as pickle_file:
                data = pkl.load(pickle_file)
            return data
        else:
            return pd.read_pickle(self._file_name)
