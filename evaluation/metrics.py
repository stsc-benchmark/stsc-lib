from enum import Enum
from typing import Callable, List, Union

import numpy as np


class MetricTarget(Enum):
    TRAJECTORY = 0
    DISTRIBUTION = 1


class Metric(object):
    def __init__(self, name, abbreviation, target: MetricTarget, scalar_res: bool, fun: Callable[[List, List], Union[List[float], float]]):  # Callable[..., float]
        self.name = name
        self.abbreviation = abbreviation
        self.target = target
        self.scalar_res = scalar_res
        self._fun = fun

    def __call__(self, y_model: List, y_gt: List) -> Union[List[float], float]:
        return self._fun(y_model, y_gt)


class _MetricFunctions(object):
    @staticmethod
    def avg_displacement_error(y_model: Union[np.ndarray, List[np.ndarray]], y_gt: Union[np.ndarray, List[np.ndarray]]) -> List[float]:  # return error per sequence
        # fix input shape if there is only a single sample given -> target shape: [ n_seqs, seq_len, 2 ] or list of trajectories of potentially varying length
        if isinstance(y_model, np.ndarray) and len(y_model.shape) == 2:
            y_model = np.expand_dims(y_model, 0)
        if isinstance(y_gt, np.ndarray) and len(y_gt.shape) == 2:
            y_model = np.expand_dims(y_model, 0)

        assert len(y_model) == len(y_gt), f"Incompatible number of samples in y_model and y_gt ({len(y_model)} and {len(y_gt)})"
        assert np.count_nonzero(np.array([len(sample) for sample in y_model]) - np.array([len(sample) for sample in y_gt])) == 0, f"Sequence length mismatch in given data."

        # use loop as sequence length might vary (-> ill-shaped numpy array)
        errs = []
        for i in range(len(y_model)):
            errs.append(float(np.mean(np.sum(np.square(y_model[i] - y_gt[i]), axis=1), axis=0)))  # TODO: die funktion hier ist nicht korrekt

        return errs

    #@staticmethod
    #def folded_ade(y_model: List, y_gt: List) -> float:
    #    return float(np.mean(_MetricFunctions.avg_displacement_error(y_model, y_gt)))


class Metrics(Enum):  # TODO:: alls metrics are per-sample (= per-trajectory)
    ADE = Metric("Average Displacement Error", "ADE", MetricTarget.TRAJECTORY, False, _MetricFunctions.avg_displacement_error)
    #Folded_ADE = Metric("Folded Average Displacement Error", "F_ADE", MetricTarget.TRAJECTORY, True, _MetricFunctions.folded_ade)
