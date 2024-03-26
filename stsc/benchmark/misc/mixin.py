from typing import List

import numpy as np
import torch

from stsc.benchmark.misc.helpers import list_pop


class BatchGenMixin:
    def build_batch_generator(self, data: List[np.ndarray], batch_size: int, seq_len: int):
        # generator randomly selects sub trajs from a pre-calculated pool. as soon as possibilities are exhausted reset the list of available trajs
        sliced_data = []
        for traj in data:
            if len(traj) < seq_len:
                continue
            for i in range(len(traj) - seq_len + 1):
                sliced_data.append(traj[i:i+seq_len])
        sliced_data = np.asarray(sliced_data)
        all_indices = list(range(len(sliced_data)))
        available_inds = []
        while True:
            iter_reset = False
            if len(available_inds) < batch_size:
                available_inds = np.random.permutation(all_indices[:]).tolist()
                iter_reset = True

            inds = list_pop(available_inds, batch_size)
            yield torch.tensor(sliced_data[inds]).float().to(self.device), iter_reset