from typing import List, Tuple, Union, Type, TypeVar

import numpy as np

from utils.dataset import Datasets, Dataset
from evaluation.properties import Tasks, LengthMode, ProcessingMode, Target
from utils.dataset import Dataset, DatasetTasks, DatasetTargets


T = TypeVar('T', bound='TestCase')


# TODO: TestCaseCollection -> get testcase instances via this to prevent multiple copies of datasets being in memory?


class TestCase:
    @staticmethod
    def all_test_cases():
        return [TestCase.compose(dataset, task, lm, pm, target) for dataset in Datasets for task in Tasks for lm in LengthMode for pm in ProcessingMode for target in Target]  # TODO: add condition of modes are compatible

    def __init__(self, dataset: Datasets, task: Tasks, length_mode: LengthMode, processing_mode: ProcessingMode, target: Target, name: Union[None, str]):
        assert task.name in DatasetTasks[dataset.name], f"Chosen task <{task.name}> is incompatible with dataset <{dataset.name}>. Available tasks are {', '.join(DatasetTasks[dataset.name])}."
        assert target.name in DatasetTargets[dataset.name], f"Chosen target <{target.name}> is incompatible with dataset <{dataset.name}>. Available targets are {', '.join(DatasetTargets[dataset.name])}."

        self._dataset = Dataset(dataset.name, dataset.value)
        self._task = task
        self._length_mode = length_mode
        self._processing_mode = processing_mode
        self._target = target

        self._limits = None
        self.dataset_name = dataset.name
        self.task_name = task.name
        self.name = name
        if self.name is None:
            self.name = self.uuid

    @classmethod
    def compose(cls: Type[T], dataset: Datasets, task: Tasks, length_mode: LengthMode, processing_mode: ProcessingMode, target: Target, name: str = None) -> T:  # TODO: just a semantically more intuitive way of creating a TestCase instance
        return cls(**{k: v for k, v in locals().items() if k != "cls"})  # TODO: unpack as kwargs (-> use **)

    @classmethod
    def from_uuid(cls: Type[T], uuid: str) -> T:
        dataset_name, task_name, length_mode_name, processing_mode_name, target_name = uuid.split("-")
        return cls(
            dataset=Datasets[dataset_name],
            task=Tasks[task_name],
            length_mode=LengthMode[length_mode_name],
            processing_mode=ProcessingMode[processing_mode_name],
            target=Target[target_name],
            name=None
        )

    @property
    def uuid(self) -> str:
        # TODO: generate unique identifier for every possible test case -> we need baseline result-data for every checkpoint, just use uid in the file_name to organize these => also create possibility to create a testcase from its uid
        #  -> allows benchmark class to emit/iterate all test cases from just the uids => uids auch auf webseite zur verfügung stellen,w enn man menge von uids durchrattern will
        return f"{self._dataset.name}-{self._task.name}-{self._length_mode.name}-{self._processing_mode.name}-{self._target.name}"

    @property
    def window_size(self):
        if self._processing_mode is ProcessingMode.RECURRENT:
            return 1
        elif self._processing_mode is ProcessingMode.FULL:
            return self._dataset.max_representation_length
        else:
            return self._dataset.min_observation_length

    @property
    def plot_limits(self) -> List[List[float]]:
        if self._limits is None:
            data = np.concatenate(self._dataset.noisy_training_data, axis=0)
            data = np.vstack([data, np.concatenate(self._dataset.noisy_test_data, axis=0)])
            min_x = np.min(data[:, 0])
            max_x = np.max(data[:, 0])
            min_y = np.min(data[:, 1])
            max_y = np.max(data[:, 1])
            halo_x = max(1., 1.1 * (max_x - min_x) - (max_x - min_x))
            halo_y = max(1., 1.1 * (max_y - min_y) - (max_y - min_y))
            self._limits = [[min_x - halo_x, max_x + halo_x], [min_y - halo_y, max_y + halo_y]]
        return self._limits

    def get_raw_training_data(self) -> Tuple[np.ndarray, np.ndarray, List, int, int]:  # TODO: just returns all trajectories with respective ground-truth (noise-gt or gt-gt) -> x-y pair
        # returns input_set, output_set (never returns the ground-truth distribution, as everything should be learned from samples), max_repr_len, min_obs_len, max_obs_len
        min_repr_len = self._dataset.min_observation_length if self._length_mode is LengthMode.VARIABLE else self._dataset.max_representation_length
        max_repr_len = self._dataset.max_representation_length
        min_obs_len = self._dataset.min_observation_length if self._length_mode is LengthMode.VARIABLE else self._dataset.default_observation_length
        max_obs_len = self._dataset.max_observation_length if self._length_mode is LengthMode.VARIABLE else self._dataset.default_observation_length

        if self._task is Tasks.REPRESENTATION:
            return self._dataset.smooth_training_data, self._dataset.smooth_training_data, self._dataset.training_trajectory_lengths, min_repr_len, max_repr_len
        elif self._task is Tasks.DENOISING:
            return self._dataset.noisy_training_data, self._dataset.smooth_training_data, self._dataset.training_trajectory_lengths, min_repr_len, max_repr_len
        elif self._task is Tasks.SMOOTH_PREDICTION:
            return self._dataset.noisy_training_data, self._dataset.smooth_training_data, self._dataset.training_trajectory_lengths, min_obs_len, max_obs_len
        else:  # NOISE_PREDICTION
            return self._dataset.noisy_training_data, self._dataset.noisy_training_data, self._dataset.training_trajectory_lengths, min_obs_len, max_obs_len

    def get_raw_test_data(self):  # TODO: just returns all trajectories with respective ground-truth (noise-gt or gt-gt) -> x-y pair
        # returns input_set, output_set (never returns the ground-truth distribution, as everything should be learned from samples), max_repr_len, min_obs_len, max_obs_len
        min_repr_len = self._dataset.min_observation_length if self._length_mode is LengthMode.VARIABLE else self._dataset.max_representation_length
        max_repr_len = self._dataset.max_representation_length
        min_obs_len = self._dataset.min_observation_length if self._length_mode is LengthMode.VARIABLE else self._dataset.default_observation_length
        max_obs_len = self._dataset.max_observation_length if self._length_mode is LengthMode.VARIABLE else self._dataset.default_observation_length

        if self._task is Tasks.REPRESENTATION:
            return self._dataset.smooth_test_data, self._dataset.smooth_test_data, self._dataset.test_trajectory_lengths, min_repr_len, max_repr_len
        elif self._task is Tasks.DENOISING:
            return self._dataset.noisy_test_data, self._dataset.smooth_test_data, self._dataset.test_trajectory_lengths, min_repr_len, max_repr_len
        elif self._task is Tasks.SMOOTH_PREDICTION:
            return self._dataset.noisy_test_data, self._dataset.smooth_test_data, self._dataset.test_trajectory_lengths, min_obs_len, max_obs_len
        else:  # NOISE_PREDICTION
            return self._dataset.noisy_test_data, self._dataset.noisy_test_data, self._dataset.test_trajectory_lengths, min_obs_len, max_obs_len

    def eval_data(self):
        if self._task in [Tasks.SMOOTH_PREDICTION, Tasks.NOISE_PREDICTION]:
            return self._build_pred_list(**self._prepare_eval_data())
        return self._build_repr_list(**self._prepare_eval_data())

    def eval_output(self):
        data = self.eval_data()
        return [np.array(o) for _, o in data]

    def eval_data_iterator(self):  # returns an actual iterator that returns one trajectory iterator at a time, which traverses the trajectory using the given mode
        if self._task in [Tasks.SMOOTH_PREDICTION, Tasks.NOISE_PREDICTION]:
            return iter(self._build_pred_list(**self._prepare_eval_data()))
        return iter(self._build_repr_list(**self._prepare_eval_data()))

        """in_data, out_data, actual_lens, min_in_len, max_in_len = self.get_raw_test_data()

        traj_indices, obs_lens, pred_lens = [], [], []
        for i in range(len(in_data)):
            for obs_len in range(min_in_len, max_in_len + 1):
                traj_indices.append(i)
                obs_lens.append(obs_len)
                pred_lens.append(len(in_data[i]) - obs_len)

        kwargs = {
            "in_data": in_data,
            "out_data": out_data,
            "traj_indices": traj_indices,
            "obs_lens": obs_lens,
            "pred_distr": self._target is Target.DISTRIBUTION
        }

        if self._task in [Tasks.SMOOTH_PREDICTION, Tasks.NOISE_PREDICTION]:
            kwargs["pred_lens"] = pred_lens
            return self._build_pred_iterator(**kwargs)
        return self._build_repr_iterator(**kwargs) """ # TODO: wie macht man das am besten mit den verschiedenen sequenzverarbeitungsmodi? (sliding window und so?) -> nutze für sliding window immer eine fenstergröße von min_obs_len (oder 3 oder sowas)
    # TODO: -> das sollte für jede trajektorie noch einen iterator mitliefern

    def _prepare_eval_data(self):
        in_data, out_data, actual_lens, min_in_len, max_in_len = self.get_raw_test_data()

        traj_indices, obs_lens, pred_lens = [], [], []
        for i in range(len(in_data)):
            for obs_len in range(min_in_len, max_in_len + 1):
                traj_indices.append(i)
                obs_lens.append(obs_len)
                pred_lens.append(len(in_data[i]) - obs_len)

        kwargs = {
            "in_data": in_data,
            "out_data": out_data,
            "traj_indices": traj_indices,
            "obs_lens": obs_lens,
            "pred_distr": self._target is Target.DISTRIBUTION
        }

        if self._task in [Tasks.SMOOTH_PREDICTION, Tasks.NOISE_PREDICTION]:
            kwargs["pred_lens"] = pred_lens

        return kwargs

    def _build_repr_list(self, in_data: np.ndarray, out_data: np.ndarray, traj_indices: List[int], obs_lens: List[int], pred_distr):
        if pred_distr:  # TODO: das ist noch falsch
            return [
                (in_data[i][:obs_lens[i]], self._dataset.partitioned_conditional_joint_distribution(in_data[i][:obs_lens[i]])) for i in traj_indices
            ]

        return [
            (in_data[i][:obs_lens[i]], out_data[i][:obs_lens[i]]) for i in traj_indices
        ]

    """def _build_repr_iterator(self, in_data: np.ndarray, out_data: np.ndarray, traj_indices: List[int], obs_lens: List[int], pred_distr):
        if pred_distr:  # TODO: das ist noch falsch
            return iter([
                (in_data[i][:obs_lens[i]], self._dataset.partitioned_conditional_joint_distribution(in_data[i][:obs_lens[i]])) for i in traj_indices
            ])

        return iter([
            (in_data[i][:obs_lens[i]], out_data[i][:obs_lens[i]]) for i in traj_indices
        ])"""

    def _build_pred_list(self, in_data: np.ndarray, out_data: np.ndarray, traj_indices: List[int], obs_lens: List[int], pred_lens: List[int], pred_distr):
        if pred_distr:
            return [
                (in_data[i][:obs_lens[i]], self._dataset.partitioned_conditional_joint_distribution(in_data[i][:obs_lens[i]])) for i in traj_indices
            ]

        return [
            (in_data[i][:obs_lens[i]], out_data[i][obs_lens[i]:obs_lens[i]+pred_lens[i]]) for i in traj_indices
        ]

    """def _build_pred_iterator(self, in_data: np.ndarray, out_data: np.ndarray, traj_indices: List[int], obs_lens: List[int], pred_lens: List[int], pred_distr):  # TODO: wie macht man das wenn target eine verteilung ist? -> es muss auch obs_len bekannt sein, sodass man die ausrechnen kann => man muss in den dataset meta-daten irgenwdie speichern was die min/max obs_len ist und bei fixer länge einfach die mitte nehmen
        if pred_distr:
            return iter([
                (in_data[i][:obs_lens[i]], self._dataset.partitioned_conditional_joint_distribution(in_data[i][:obs_lens[i]])) for i in traj_indices
            ])

        return iter([
            (in_data[i][:obs_lens[i]], out_data[i][obs_lens[i]:obs_lens[i]+pred_lens[i]]) for i in traj_indices
        ])"""
