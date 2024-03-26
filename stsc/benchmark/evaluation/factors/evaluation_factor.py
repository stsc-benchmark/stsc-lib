from __future__ import annotations
from typing import List, Generator, Dict, Optional
import os
import pickle as pkl

import numpy as np

from stsc.datagen.trajectory_gmm import TrajectoryGMM
from stsc.datagen.sequence_sample_set import SequenceSampleSet
from stsc.benchmark.evaluation.metrics import Metrics
from stsc.benchmark.misc.print_writer import PrintTarget, PrintWriter
from stsc.benchmark.evaluation.factors.evaluation_factor_util import TestDataset, ResultsInfoPlotOptions


class EvaluationFactor:
    """
    Base class for factors considered during evaluation.
    Defines ground truth training and test datasets and provides basic functionality for iteration.
    Respective performance measures are defined in each sub-class.
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self._stsc_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
        self._data_gens: Dict[str,TrajectoryGMM] = {}
        self._training_datasets: Dict[str, List[np.ndarray]] = {}
        self._test_datasets: Dict[str, List[TestDataset]] = {}
        self._cur_train_iter_key: Optional[str] = None
        self._cur_test_iter_index: Optional[int] = None

    def observation_lengths(self, training_dataset_name: str) -> List[int]:
        raise Exception("Function <observation_lengths> not implemented for subclass.")

    def prediction_lengths(self, training_dataset_name: str) -> List[int]:
        raise Exception("Function <prediction_lengths> not implemented for subclass.")

    @property
    def data_distributions(self) -> Dict[str,TrajectoryGMM]:
        return self._data_gens

    def data_distribution(self, data_generator_name: str) -> TrajectoryGMM:
        return self._data_gens[data_generator_name]

    @property
    def training_datasets(self) -> Dict[str,List[np.ndarray]]:
        return self._training_datasets

    @property
    def training_datasets_names(self) -> List[str]:
        return list(self._training_datasets.keys())

    def test_datasets(self, training_dataset_name: str) -> List[List[np.ndarray]]:
        return self._test_datasets[training_dataset_name]

    def test_dataset_names(self, training_dataset_name: str) -> List[str]:
        return [td.name for td in self._test_datasets[training_dataset_name]]

    @property
    def train_iter_key(self) -> Optional[str]:
        return self._cur_train_iter_key

    @property
    def test_iter_index(self) -> Optional[int]:
        return self._cur_test_iter_index

    def iter_test_datasets(self, training_dataset_name: str) -> Generator[TestDataset]:
        for i in range(len(self._test_datasets[training_dataset_name])):
            self.set_test_iter_index(i)
            yield self._test_datasets[training_dataset_name][i]

    def set_test_iter_index(self, i: int) -> None:
        assert 0 <= i < len(self._test_datasets[self._cur_train_iter_key]), f"Given index {i} must be within [0, {len(self._test_datasets[self._cur_train_iter_key])})."
        self._cur_test_iter_index = i

    def iter_training_datasets(self) -> Generator[Dict[str,List[np.ndarray]]]:  
        for key, val in self._training_datasets.items():
            self.set_train_iter_key(key)
            yield key, val

    def set_train_iter_key(self, key: str) -> None:
        assert key in self._training_datasets, f"Given key {key} must be in {self._training_datasets.keys()}."
        self._cur_train_iter_key = key

    def evaluate_metrics(
        self, 
        prediction_samples: Dict[int,SequenceSampleSet], 
        print_writer: Optional[PrintWriter] = None, 
        results_info_plots_options: Optional[ResultsInfoPlotOptions] = None, 
        model_name: Optional[str] = None
    ) -> Dict[int, Dict[Metrics,float]]:
        """ 
        Function for running through all performance measures defined in respective sub-classes.

        Caution: train_iter_key and test_iter_index values have to be set prior to calling this function. These are set automatically when using the provided generators (<iter_training_datasets> and <iter_test_datasets>) 
        """
        raise Exception("Function <evaluate_metrics> not implemented for subclass.")

    def _dump_datasets(self, file_path) -> None:
        with open(file_path, "wb") as f:
            pkl.dump({
                dgname: {
                    "ds_gmm": {"weights": self._data_gens[dgname].weights, "means": self._data_gens[dgname].means, "covs": self._data_gens[dgname].covs, "sampling_seed": self._data_gens[dgname].sampling_seed, "name": dgname},
                    "sampled_train_dataset": self._training_datasets[dgname],
                    "sampled_test_datasets": self._test_datasets[dgname]
                } for dgname in self._data_gens.keys()
            }, f)

    def _load_datasets(self, file_path) -> None:
        self._data_gens.clear()
        self._training_datasets.clear()
        self._test_datasets.clear()
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            for dgname in data.keys():
                self._data_gens[dgname] = TrajectoryGMM(**data[dgname]["ds_gmm"])
                self._training_datasets[dgname] = data[dgname]["sampled_train_dataset"]
                self._test_datasets[dgname] = data[dgname]["sampled_test_datasets"]

    @classmethod
    def _setup_print_writer(cls, pw: Optional[PrintWriter]) -> PrintWriter:
        if pw is None:
            return PrintWriter((PrintTarget.NONE, None))
        return pw

    @classmethod
    def _setup_plot_dir(cls, plot_options: Optional[ResultsInfoPlotOptions]) -> None:
        if plot_options is None:
            return
        if not os.path.exists(plot_options.directory_path):
            os.makedirs(plot_options.directory_path)
