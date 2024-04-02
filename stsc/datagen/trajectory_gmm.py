from __future__ import annotations
import functools
from typing import List, Optional, Union
import pickle as pkl
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from stsc.datagen.gmm_sequence import GMMSequence
from stsc.datagen.common import confidence_ellipse


class TrajectoryGMM: 
    """
    This class represents a Gaussian mixture for 2D trajectories of pre-defined lengths.
    In that, the mixture basically represents a Gaussian process prior of a Bézier Spline Gaussian process.
    The mixture parameters are calculated via the mean and kernel functions of this GP, where each component then consists of a mean vector and the gram matrix computed with the kernel function.
    Each Gaussian component thereby covers a trajectory of length n, where the mean vector concatenates the 2D mean vectors (e.g. [x1,y1,x2,y2,...xn,yn]) of all time steps and the gram matrix is a block-partitioned matrix.
    In this way, correlations between trajectory points can be modeled effectively.
    Note: Each component can cover trajectories of different length (e.g. for covering paths of different in a static scene)

    The class is build as a state machine, that can switch between its prior (initial) state and posterior states calculated under the presence of data (observed trajectories in this case). 
    """
    def __init__(
        self, 
        weights: Union[List, np.ndarray], 
        means: List[np.ndarray], 
        covs: List[np.ndarray],
        name: Optional[str] = None
    ) -> None:
        """ 
        Initializer notes:
          <means> must contain flattened mean sequence vectors (of arbitrary length). 
          <covs> is a block-partitioned matrix.

        :param weights:
        :param means:
        :param covs:
        :param sampling_seed: seed used for <sample> function
        """

        if type(weights) is list:
            weights = np.asarray(weights)

        self.n_comps = len(weights)

        assert len(means) == self.n_comps and len(covs) == self.n_comps, "Mismatch in number of components in either means or covs."

        # unconditioned params
        self._prior_weights = np.copy(weights)
        self._prior_means = means[:]
        self._prior_covs = covs[:]  
        self._active_components = np.ones(self.n_comps).astype(bool)

        self.weights = weights
        self.means = means
        self.covs = covs
        self.seq_lengths = [len(means[k]) // 2 for k in range(self.n_comps)]
        self._cond_covars = None
        self._cond_sequence = None
        self._cond_indices: Optional[List[int]] = None
        self._cond_start_indices: List[int] = self.n_comps * [0]
        self.name = name

    def save(self, save_file_path: str) -> None:
        with open(save_file_path, "wb") as sf:
            pkl.dump({
                "weights": self.weights,
                "means": self.means,
                "covs": self.covs,
                "name": self.name
            }, sf)

    @classmethod
    def from_file(cls, file_path: str) -> TrajectoryGMM:
        assert os.path.exists(file_path), "Couldn't create <TrajectoryGMM> from file. File does not exist."
        with open(file_path, "rb") as sf:
            return cls(**pkl.load(sf))

    def sequence_distribution(self, exclude_condition_indices: bool = True) -> GMMSequence:
        """ 
        Returns the (conditional) probability distribution in terms of a <GMMSequence>, after putting this TrajectoryGMM into posterior mode. 
        In prior mode, just returns the full prior distribution as <GMMSequence>
        """
        gt_weights = []
        gt_means = []
        gt_covs = []
        for k in range(self.n_comps):
            if not self._active_components[k]:
                continue

            # TODO: add possibility to include condition indices
            target_indices = list(set(range(self.seq_lengths[k])) - set(list(range(self._cond_start_indices[k])) + list(self._cond_indices[k])))
            gt_weights.append(self.weights[k])
            m, c = self._marginal_normal(self.means[k], self.covs[k], target_indices)

            m = np.reshape(m, [-1, 2])
            c = np.array([c[2*i:2*i+2, 2*i:2*i+2] for i in range(c.shape[0] // 2)])
            
            gt_means.append(m)
            gt_covs.append(c)
        
        return GMMSequence(gt_weights, gt_means, gt_covs)

    def posterior(
        self, 
        observation: np.ndarray, 
        #obs_indices: Union[np.ndarray,List[int]], 
        consider_pred_len: Optional[int] = None
    ) -> TrajectoryGMM:
        """ 
        Switches dataset into posterior distribution mode conditioned on given observation (with shape [ trajectory length, 2 ]). 

        Technical note: For each mixture components seeks for the len(observation) length sub-sequence that is closest to the given observation in terms of the euclidean distance. 
        Additional Note: Although it is theoretically possible to calculate the posterior considering all components, for numerical reasons, it is however adviced to provide the prediction length, that the posterior will be used for, in the case of datasets with a large number of components. Usually this will allow to ignore several components during posterior calculation, keeping floats more within a reasonable range.

        Partitions mean and covariance matrices like so:

        mu = / mu_a \   cov = / c_aa c_ab \
             \ mu_b /         \ c_ba c_bb /,

        where partition 'a' contains the observed steps.

        The conditional Gaussian are given by:

        mu_b|a = mu_b + c_ba c^-1_aa (x_a - mu_a)
        c_b|a = c_bb - c_ba c^-1_aa c_ab

        On a per-component basis, the posterior weight distribution is given by:
        
        w'_k = ( w_k * N(x_a | mu_ka, c_kaa) ) / ( sum_k [ w_k * N(x_a | mu_ka, c_kaa) ] ), 
        
        where N(x|m,c) is the Gaussian distribution.

        Note: After calculating the posterior (which has a lower dimension than the prior), the posterior mean vector and covariance matrix is inflated to the original dimensionality to allow more straightforward/consistent indexing.
        This is done by inserting the observed points into the mean vector at the respective positions and blocks of zeros are added into the covariance matrix.
        For unused indices < start_index (= min(obs_indices)), the mean vector and covariance matrix is filled with zeros.

        :param observation: observed trajectory
        :param obs_indices: indices for associating observed trajectory points with indices of the Gaussian process modeled by this object's mixture.
        :param consider_pred_len: prediction length to consider within which the posterior will be used
        """
        assert len(observation.shape) == 2, "Observation must be a sequence of trajectory points."
        assert observation.shape[-1] == 2, "Observation trajectory point dimension must be 2."
        #assert len(obs_indices) == len(observation), f"Number of sequences indices ({len(obs_indices)}) must match number of given observed points ({len(observation)})."

        #if type(obs_indices) is list:
        #    obs_indices = np.asarray(obs_indices)

        # Determine closest sub-sequence for each mixture component
        flat_obs = observation.reshape([-1])
        obs_indices = np.empty([self.n_comps, len(observation)], dtype=int)
        for k in range(self.n_comps):
            sub_means = np.array([self._prior_means[k][i:i+len(flat_obs)] for i in range(0, len(self._prior_means[k]) - len(flat_obs) + 1, 2)])  # we are in 2d -> step 2
            #sub_covs = np.array([self.covs[k][i:i+len(flat_obs), i:i+len(flat_obs)] for i in range(0, len(self.means[k]) - len(flat_obs) + 1, 2)])
            dists = [np.linalg.norm(m - flat_obs) for m in sub_means]
            #mahalanobis = [np.sqrt((sub_means[i] - flat_obs).T @ np.linalg.inv(sub_covs[i]) @ (sub_means[i] - flat_obs)) for i in range(len(sub_means))]
            closest_subseq_start = np.argmin(dists)
            obs_indices[k] = np.arange(start=closest_subseq_start, stop=closest_subseq_start + len(observation))

        start_indices = [obs_inds[0] for obs_inds in obs_indices]
        self._active_components[:] = True

        # re-organize components to comply with observed indices
        # 1. "deactivate" components where observed indices exceeed component's sequence length
        if consider_pred_len:
            for k in range(self.n_comps):
                self._active_components[k] = len(self._prior_means[k]) // 2 >= obs_indices[k][-1] + consider_pred_len

        # 2. re-normalize weights (set inactive component's weight to 0)
        ws = np.sum([w for k, w in enumerate(self._prior_weights) if self._active_components[k]])
        renorm_weights = np.array([w / ws if self._active_components[k] else 0. for k, w in enumerate(self._prior_weights)])

        #flat_obs = observation.reshape([-1])
        w_cond_norm = np.sum([renorm_weights[k] * self._normal_pdf(*self._marginal_normal(self._prior_means[k],
                                                                                          self._prior_covs[k],
                                                                                          obs_indices[k]), 
                                                                    x=flat_obs) 
                              for k in range(self.n_comps) if self._active_components[k]])
       
        cond_weights, cond_means, cond_covs = [], [], []
        self._cond_covars = []
        obs_ind = list(range(len(obs_indices[k])))  # from here we will ignore vector/matrix indices < start_index, our observation then starts at index 0
        for k in range(self.n_comps):
            if not self._active_components[k]:
                cond_weights.append(0.)
                cond_means.append(np.zeros(2))
                cond_covs.append(np.zeros([2, 2]))
                self._cond_covars.append(np.zeros([2, 2]))
                continue
            
            w = renorm_weights[k]
            m = self._prior_means[k][2*start_indices[k]:]
            c = self._prior_covs[k][2*start_indices[k]:, 2*start_indices[k]:]
            mm, mc = self._marginal_normal(m, c, obs_ind)
            cm, cc, p_inds = self._conditional_normal(m, c, obs_ind, flat_obs, ret_permutation=True)

            # Inflate conditional mean vector and covariance matrix to original shape by inserting
            # a) observations into the mean vector
            # b) zeros into the covariance matrix
            # This preservation of shape helps with indexing later on (which is its sole purpose)
            # Note: p_inds contains the observed indices first, followed by the rest (both in ascending order)
            inv_p_inds = [p_inds.index(v) for v in range(len(p_inds))]
            inflated_cm = np.reshape(np.hstack([flat_obs, cm]).reshape([-1, 2])[inv_p_inds], [-1])  # apply inverse permutation
            inflated_cm = np.hstack([np.zeros(2*start_indices[k]), inflated_cm])  # add skipped indices

            n_add = 2 * len(obs_indices[k])
            tmp = np.row_stack([np.zeros(shape=[n_add, cc.shape[1]]), cc])
            tmp = np.column_stack([np.zeros(shape=[n_add + cc.shape[0], n_add]), tmp])
            inflated_cc = np.zeros(shape=tmp.shape)
            for i, ci in enumerate(inv_p_inds):
                for j, cj in enumerate(inv_p_inds):
                    inflated_cc[2*i:2*i+2, 2*j:2*j+2] = tmp[2*ci:2*ci+2, 2*cj:2*cj+2]
            inflated_cc2 = np.zeros(shape=[inflated_cc.shape[0] + 2*start_indices[k], inflated_cc.shape[1] + 2*start_indices[k]])
            inflated_cc2[2*start_indices[k]:, 2*start_indices[k]:] = inflated_cc

            pdf_scale = self._normal_pdf(mm, mc, flat_obs)
            cond_weight = (w * pdf_scale) / w_cond_norm
            
            cond_weights.append(cond_weight)
            cond_means.append(inflated_cm)
            cond_covs.append(inflated_cc2) 
            self._cond_covars.append(cc)
        
        self.weights = np.array(cond_weights)
        self.means = cond_means
        self.covs = cond_covs
        self._cond_sequence = np.copy(observation)
        self._cond_indices = obs_indices[:]
        self._cond_start_indices = start_indices[:]

        return self

    def prior(self) -> TrajectoryGMM:
        """ Switches Dataset into prior distribution mode (reverts any conditioning). """
        self.weights = np.copy(self._prior_weights)
        self.means = np.copy(self._prior_means)
        self.covs = np.copy(self._prior_covs)
        self._active_components[:] = True
        self._cond_covars = None
        self._cond_sequence = None
        self._cond_indices = None
        self._cond_output_indices = None
        self._cond_start_indices = self.n_comps * [0]
        return self

    def sample(self, num_samples: int, cap_length: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]: 
        """ 
        Draws <num_samples> samples from the prior or posterior GP distribution.
        Sampled trajectories are reshaped into 2D sequences.
        """
        # TODO: possibility to restrict samples to 1 and 2 sigma regions only (to prevent strong outliers)
        # TODO: use sample_component internally
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        for _ in range(num_samples):
            k = rng.choice(self.n_comps, p=self.weights)
            si = 2*self._cond_start_indices[k]
            s = rng.multivariate_normal(self.means[k][si:], self.covs[k][si:, si:])
            s = s.reshape([-1, 2])
            if cap_length is not None:
                s = s[:cap_length]
            samples.append(s)

        return samples

    def sample_component(
            self, 
            component: int, 
            num_samples: int, 
            cap_length: Optional[int] = None, 
            scale_variance: Optional[float] = None,
            rng: Optional[np.random.Generator] = None
        ) -> List[np.ndarray]:
        """
        Draws <num_samples> samples from the prior or posterior GP distribution considering only the given component.
        Sampled trajectories are reshaped into 2D sequences. 
        
        By providing a <scale_variance> value, samples are restricted to an ellipctical region around the mean vector, which conceptually follows to be somewhat comparable to the limit_variance * sigma region around the mean in the univariate case. This is achieved by just scaling the covariance matrix.
        """
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        si = 2*self._cond_start_indices[component]
        if scale_variance is not None:
            cov = self.covs[component][si:, si:] * scale_variance
        else:
            cov = self.covs[component][si:, si:]

        for _ in range(num_samples):
            s = rng.multivariate_normal(self.means[component][si:], cov)
            s = s.reshape([-1, 2])
            if cap_length is not None:
                s = s[:cap_length]
            samples.append(s)

        return samples

    def plot(
        self, 
        n_steps: Optional[int] = None, 
        draw_prior_mean: bool = False,
        axis: Optional[plt.Axes] = None, 
        label_prefix: str = "", 
        draw_stds: List = [3, 2, 1], 
        suppress_alpha: bool = False,
        legend=True,
        title=None,
        show: bool = False
    ) -> None:
        """
        Draws this GP into a given pyplot axis according to the current state (prior or posterior). 
        """

        if axis is None:
            axis = plt.gca()

        if self._cond_sequence is not None:
            axis.plot(self._cond_sequence[:, 0], self._cond_sequence[:, 1], "kd", alpha=0.75)
            if draw_prior_mean:
                for k in range(self.n_comps):
                    m_k = self._prior_means[k].reshape(-1, 2)
                    axis.plot(m_k[:, 0], m_k[:, 1], "x--", color="gray", alpha=0.5)

            alpha_factors = [1., 1., 1.] if suppress_alpha else [0.33, 0.66, 1.]
            for comp in range(self.n_comps):
                if not self._active_components[comp] or self.weights[comp] < 1e-6:
                    continue

                mseq = self.means[comp].reshape(-1, 2)
                draw_inds = list(set(range(len(mseq))) - set(list(range(self._cond_start_indices[comp])) + list(self._cond_indices[comp])))
                if n_steps is not None:
                    draw_inds = draw_inds[:n_steps]
                alpha = 1. if suppress_alpha else self.weights[comp]

                lines = axis.plot(mseq[draw_inds, 0], mseq[draw_inds, 1], "o", alpha=alpha, label=f"{label_prefix}{self.weights[comp]:.2f}")
                cur_color = lines[0].get_color()

                for i in draw_inds:
                    step_cov = self.covs[comp][2*i:2*i+2, 2*i:2*i+2]
                    confidence_ellipse(mseq[i], step_cov, axis, 3, cur_color, "none", alpha*0.33)
                    confidence_ellipse(mseq[i], step_cov, axis, 2, cur_color, "none", alpha*0.66)
                    confidence_ellipse(mseq[i], step_cov, axis, 1, cur_color, "none", alpha*1.)

            plt.title(title)
            if legend:
                plt.legend()
            if show:
                plt.show()
            return

        for comp in range(self.n_comps):
            mseq = self.means[comp].reshape(-1, 2)
            alpha = self.weights[comp]

            lines = axis.plot(mseq[:, 0], mseq[:, 1], "o-", alpha=alpha, label=f"{self.weights[comp]:.2f}")
            cur_color = lines[0].get_color()
            axis.plot(mseq[:1, 0], mseq[:1, 1], "o", color=cur_color, markerfacecolor="none", markersize=10)

            for t in range(self.seq_lengths[comp]):
                step_cov = self.covs[comp][2*t:2*t+2, 2*t:2*t+2]
                confidence_ellipse(mseq[t], step_cov, axis, 3, cur_color, "none", alpha*0.33)
                confidence_ellipse(mseq[t], step_cov, axis, 2, cur_color, "none", alpha*0.66)
                confidence_ellipse(mseq[t], step_cov, axis, 1, cur_color, "none", alpha*1.)

        plt.title(title)
        if legend:
            plt.legend()
        if show:
            plt.show()

    def plot_samples(
        self, 
        num_samples: int = 10, 
        draw_modes: bool = True, 
        axis: Optional[plt.Axes] = None, 
        show: bool = False
    ) -> None:
        """ Draws <num_samples> samples and draws these to given pyplot axis. """
        if axis is None:
            axis = plt.gca()

        if draw_modes:
            for comp in range(self.n_comps):
                mseq = self.means[comp][2*self._cond_start_indices[comp]:].reshape(-1, 2)
                axis.plot(mseq[:, 0], mseq[:, 1], "k--", alpha=0.75)

        for sample in self.sample(num_samples):
            plt.plot(sample[:, 0], sample[:, 1], "o-")

        if show:
            plt.show()

    @staticmethod
    def _marginal_normal(mean, covar, inds):
        assert len(mean.shape) == 1 and len(covar.shape) == 2, "Only a single mean vector and covariance matrix must be given"

        inds = list(functools.reduce(lambda x,y: x+y, [[2*i, 2*i+1] for i in inds]))
        
        mean_marg = mean[inds]
        covar_marg = []
        for r in range(covar.shape[0]):
            for c in range(covar.shape[1]):
                if r in inds and c in inds:
                    covar_marg.append(covar[r, c])

        return mean_marg, np.reshape(covar_marg, [len(inds), len(inds)])

    @staticmethod
    def _conditional_normal(mean, covar, cond_inds, observation, ret_permutation=False):
        """ Calculates conditional normal distribution given an observation. """
        assert len(mean.shape) == 1 and len(covar.shape) == 2, "Only a single mean vector and covariance matrix must be given"

        other_inds = list(set(range(len(mean) // 2)).difference(set(cond_inds)))
        permute_inds = list(cond_inds) + other_inds

        rearranged_mean = mean[list(functools.reduce(lambda x,y: x+y, [[2*i, 2*i+1] for i in permute_inds]))]
        rearranged_covar = np.zeros(shape=covar.shape)
        for i, ci in enumerate(permute_inds):
            for j, cj in enumerate(permute_inds):
                rearranged_covar[2*i:2*i+2, 2*j:2*j+2] = covar[2*ci:2*ci+2, 2*cj:2*cj+2]
        
        inds_a = list(functools.reduce(lambda x,y: x+y, [[2*i, 2*i+1] for i in np.arange(len(cond_inds), dtype=np.int32)]))
        inds_b = list(functools.reduce(lambda x,y: x+y, [[2*i, 2*i+1] for i in np.arange(len(other_inds), dtype=np.int32) + len(cond_inds)]))

        m_a = rearranged_mean[inds_a]
        m_b = rearranged_mean[inds_b]
        c_aa, c_ab, c_bb = TrajectoryGMM._partition_matrix(rearranged_covar, inds_a, inds_b)
        c_ba = c_ab.T
        inv_c_aa = np.linalg.inv(c_aa)

        cond_mean = m_b + c_ba @ inv_c_aa @ (observation - m_a) 
        cond_covar = c_bb - c_ba @ inv_c_aa @ c_ab  

        # TODO: check this again: force diagonal values to be >= 1e-10 due to numerical inaccuracies
        for i in range(cond_covar.shape[0]):
            cond_covar[i, i] = np.maximum(cond_covar[i, i], 1e-10)

        if ret_permutation:
            return cond_mean, cond_covar, permute_inds
        return cond_mean, cond_covar
           
    @staticmethod
    def _partition_matrix(matrix, inds_a, inds_b):
        """ Partition a matrix into two diagonal blocks and upper off-diagonal block. """
        tmp_aa, tmp_ab, tmp_bb = [], [], []
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if r in inds_a and c in inds_a:
                    tmp_aa.append(matrix[r, c])
                elif r in inds_b and c in inds_b:
                    tmp_bb.append(matrix[r, c])
                elif r in inds_a and c in inds_b:
                    tmp_ab.append(matrix[r, c])
        c_aa = np.reshape(tmp_aa, [len(inds_a), len(inds_a)])
        c_ab = np.reshape(tmp_ab, [len(inds_a), len(inds_b)])
        c_bb = np.reshape(tmp_bb, [len(inds_b), len(inds_b)])

        return c_aa, c_ab, c_bb
    
    @staticmethod
    def _normal_pdf(mu: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:
        """ Evaluates the Gaussian probability density function for a given value x. """
        d = len(x)
        denom = np.sqrt((2*np.pi)**d * np.linalg.det(cov))
        return (1. / denom) * np.exp(-0.5 * ((x - mu).T @ np.linalg.inv(cov)) @ (x - mu))
        