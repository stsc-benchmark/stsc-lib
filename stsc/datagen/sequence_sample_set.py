from typing import List, Tuple

import numpy as np
import scipy.stats as stats


class SequenceSampleSet:
    """ Wraps a set of sample sequences and provides utility functions for accessing samples and calculated kde's for specific sequence indices. """
    def __init__(self, samples: np.ndarray) -> None:
        """ Assumed samples array shape: [ n_sequences, sequence_length, n_samples, 2 ] """
        assert len(samples.shape) == 4, "Ill-formed samples array (probably wrong shape?). Required shape: [ n_sequences, sequence_length, n_samples, 2 ]"

        self.n_sequences, self.sequence_length, self.n_samples, _ = samples.shape
        self.samples = np.copy(samples)
        self._kde_cache = {} 

    def index_samples(self, sequence_index: int) -> List[np.ndarray]:
        """ Returns sequence of samples for given sequence (by index). """
        return [self.step_samples(sequence_index, step) for step in range(self.sequence_length)]

    def index_kdes(self, sequence_index: int) -> List[stats.gaussian_kde]:
        """ Returns sequence of Gaussian KDE's for given sequence (by index). """
        return [self.step_kde(sequence_index, step) for step in range(self.sequence_length)]

    def index_mean_covs(self, sequence_index: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ Returns sequence of mean vectors and covariance matrices for given sequence (by index). """
        return [self.step_mean_cov(sequence_index, step) for step in range(self.sequence_length)]

    def step_samples(self, sequence_index: int, step: int) -> np.ndarray:
        """ Returns samples for specific step within given sequence (by index). """
        return self.samples[sequence_index, step]

    def step_kde(self, sequence_index: int, step: int) -> stats.gaussian_kde:
        """
        Calculates the a Gaussian KDE across the samples of all sequences for a specific index.
        Uses scipy.stats.gaussian_kde.
        Note: Calculated KDE's are cached. 
        """
        if not step in self._kde_cache:
            self._kde_cache[(sequence_index, step)] = stats.gaussian_kde(self.samples[sequence_index, step].transpose([1, 0]))
        return self._kde_cache[(sequence_index, step)]

    def step_mean_cov(self, sequence_index: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean vector and covariance matrix across the sapmles of all sequences for a specific index.
        Uses numpy.mean and numpy.cov.
        """
        return np.mean(self.samples[sequence_index, step], axis=0), np.cov(self.samples[sequence_index, step].T)
