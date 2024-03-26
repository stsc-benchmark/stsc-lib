from typing import List

import numpy as np


def ndarray_to_sequence_list(arr: np.ndarray) -> List[np.ndarray]:
    assert len(arr.shape) == 3, "Expected shape for <arr> is [ n_sequences, sequence_length(s), data_dim ]."
    return [arr[i] for i in range(len(arr))]


def list_pop(lst: List, n: int) -> List:
    popped = lst[:n]
    del lst[:n]
    return popped
