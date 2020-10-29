from typing import Union, List

import numpy as np


def copy_data(data: Union[List[np.ndarray], np.ndarray]):
    if isinstance(data, list):
        return [np.copy(e) for e in data]
    return np.copy(data)


def lmap(fun, lst):
    return list(map(fun, lst))


def nlmap(fun, np_arr):
    return np.asarray(list(map(fun, np_arr)))
