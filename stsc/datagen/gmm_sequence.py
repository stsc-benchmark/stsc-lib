from typing import List

import numpy as np
import scipy.stats as stats


class GMM:
    """ Simplified wrapper for a Gaussian mixture model providing basic functionality. """
    def __init__(self, weights: np.ndarray, mean_vectors: np.ndarray, covar_matrices: np.ndarray) -> None:
        """
        Initializes a GMM. Internally uses the <multivariate_normal> object from the <scipy.stats> package.
        """
        self.weights = weights
        self.mean_vectors = mean_vectors
        self.covar_matrices = covar_matrices
        self._n_comps = len(self.weights)
        self._gaussians = [stats.multivariate_normal(mean_vectors[k], covar_matrices[k]) for k in range(len(self.weights))]
        self._rng = np.random.default_rng()

    @property
    def n_comps(self) -> int:
        return self._n_comps

    def pdf(self, x: np.ndarray) -> float:
        """ Evaluates the probability density function for a given vector x. """
        return np.sum([self.weights[k] * self._gaussians[k].pdf(x) for k in range(self._n_comps)], axis=0)

    def sample(self, n_samples, rng: np.random.Generator = None) -> np.ndarray:
        """ Draws <n_samples> samples from the mixture distribution. """
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        for _ in range(n_samples):
            k = rng.choice(self._n_comps, p=self.weights)
            samples.append(rng.multivariate_normal(self.mean_vectors[k], self.covar_matrices[k]))
        return np.asarray(samples)


class GMMSequence:
    """
    Container object that stores a sequence of GMMs.
    This sequence of GMMs models sequences in terms of <n_comp> components, comprised of sequences of Gaussian random variables, with component weights applying to each entire respective sequence. 

    When accessing a GMM at a specific index, the global sequence weights are assigned to the local GMM. 
    This container supports sequences (components) of varying length. When trying to access an index that is out of bounds for a component, this component is left out and the weights are re-normalized. 
    """
    def __init__(self, weights: np.ndarray, mean_vectors: List[np.ndarray], covar_matrices: List[np.ndarray]) -> None:
        """ 
        Expected shapes:
        - weights: [ n_comps ]
        - mean_vectors: [ n_comps, seq_length, 2 ]
        - covar_matrices: [ n_comps, seq_length, 2, 2]
        """
        self.n_comps = len(weights)
        self.weights = np.copy(weights)
        self.mean_vectors = [np.copy(e) for e in mean_vectors]
        self.covar_matrices = [np.copy(e) for e in covar_matrices]

        self._gmm_cache = {}

    def __call__(self, index: int) -> GMM:
        """ Returns the GMM corresponding to the a given sequence index. """
        if index == -1:
            index = len(self.mean_vectors[0]) - 1
        if not index in self._gmm_cache:
            ws = []
            ms = []
            cs = []
            renorm = False
            for k in range(self.n_comps):
                if index >= len(self.mean_vectors[k]):  # take out component if it does not have a value for given index (not long enough)
                    renorm = True
                    continue
                ws.append(self.weights[k])
                ms.append(self.mean_vectors[k][index])
                cs.append(self.covar_matrices[k][index])
            if renorm:
                ws = ws / np.sum(ws)
            if np.any(np.isnan(ws)):
                print()
            self._gmm_cache[index] = GMM(ws, ms, cs)
        
        return self._gmm_cache[index]
