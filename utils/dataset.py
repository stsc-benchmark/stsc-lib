from enum import Enum
import os

import numpy as np
import scipy.stats as stats

import utils.io as io
from evaluation.properties import Tasks, Target


#root = f"{os.path.split(os.path.abspath(io.__file__))[0]}/.."
root = io.module_path()


class Datasets(Enum):
    TMAZE_DISTR = f"{root}/datasets/synthetic/tmaze_distr.gz"
    TMAZE_DISTR_COMP_BIAS = f"{root}/stsc-lib/datasets/synthetic/tmaze_distr_comp_bias.gz"
    TMAZE_DISTR_POS_BIAS = f"{root}/stsc-lib/datasets/synthetic/tmaze_distr_pos_bias.gz"


DatasetTasks = {
    Datasets.TMAZE_DISTR.name: [Tasks.DENOISING.name, Tasks.NOISE_PREDICTION.name, Tasks.SMOOTH_PREDICTION.name],
    Datasets.TMAZE_DISTR_COMP_BIAS.name: [Tasks.DENOISING.name, Tasks.NOISE_PREDICTION.name, Tasks.SMOOTH_PREDICTION.name],
    Datasets.TMAZE_DISTR_POS_BIAS.name: [Tasks.DENOISING.name, Tasks.NOISE_PREDICTION.name, Tasks.SMOOTH_PREDICTION.name]
}


DatasetTargets = {
    Datasets.TMAZE_DISTR.name: [Target.TRAJECTORY.name, Target.DISTRIBUTION.name],
    Datasets.TMAZE_DISTR_COMP_BIAS.name: [Target.TRAJECTORY.name, Target.DISTRIBUTION.name]
}


class Dataset(object):  # TODO: verwende load methode, die u.u. ein GMMDataset objekt zurückgibt, wenn nötig, ansonsten ein dataset objekt => geht nicht wegen lazy_loading, gmm parts müssen hier integriert sein
    def __init__(self, name: str, file_path: str, load_lazily: bool = True):  # TODO: variable die angibt ob alle trajektorien gleich lang sind oder nicht -> die info braucht man, wenn es darum geht beispiele variable/unterschiedlicher länge rauszuholen
        self.name = name
        self.file_path = file_path
        self.load_lazily = load_lazily
        self._loaded = False

        # TODO: verwende klasse ein bisschen wie einen zustandsautomaten (siehe auch unten die funktion <set_variant>
        self._raw_dataset = None
        self.is_gmm_dataset = False

        if not load_lazily:
            self._load_dataset()

    def _load_dataset(self):
        if self._loaded:
            return

        raw_data = io.decompress_data(self.file_path)

        self._raw_dataset = self._read_data(raw_data)
        if "distribution" in self._raw_dataset.keys():
            self.is_gmm_dataset = True

        self._loaded = True

    def _read_data(self, data_dict):
        data = {
            "meta": data_dict["meta"],  # "meta": {"default_obs_len": Z, "min_obs_len": X, "max_obs_len": Y, "max_repr_len": W}
            "noise-data": {
                "training": np.array(data_dict["noisy"]["train"]), "test": np.array(data_dict["noisy"]["test"])
            },
            "smooth-data": {
                "training": np.array(data_dict["smooth"]["train"]), "test": np.array(data_dict["smooth"]["test"])
            },
            "actual_traj_lens": {
                "training": data_dict["actual_traj_lens"]["train"],
                "test": data_dict["actual_traj_lens"]["test"]
            }
        }

        if "gt_dist" in data_dict.keys():
            data["distribution"] = {
                    "pi": np.array(data_dict["gt_dist"]["component_weights"]),
                    "mu": np.array(data_dict["gt_dist"]["mean_vectors"]),
                    "sigma": np.array(data_dict["gt_dist"]["covariance_matrices"])
                }

        return data
    """
    "meta": {"default_obs_len": 8, "min_obs_len": 8, "max_obs_len": 16, "max_repr_len": 30},
    "gt_dist": {
        "component_weights": [0.5, 0.5],
        "mean_vectors": [comp_mean_1.tolist(), comp_mean_2.tolist()],
        "covariance_matrices": [np.diag(cov).tolist(), np.diag(cov).tolist()]
    },
    "noisy": {
        "train": train_balanced.tolist(),
        "test": test_balanced.tolist()
    },
    "smooth": {
        "train": gt_train_balanced.tolist(),
        "test": gt_test_balanced.tolist()
    },
    "actual_traj_lens": {
        "train": [30 for _ in range(len(gt_train_balanced))],
        "test": [30 for _ in range(len(gt_test_balanced))],
    }"""

    @property
    def raw_data(self):
        self._load_dataset()
        return self._raw_dataset

    @property
    def smooth_training_data(self):
        self._load_dataset()
        return self._raw_dataset["smooth-data"]["training"]

    @property
    def noisy_training_data(self):
        self._load_dataset()
        return self._raw_dataset["noise-data"]["training"]

    @property
    def smooth_test_data(self):
        self._load_dataset()
        return self._raw_dataset["smooth-data"]["test"]

    @property
    def noisy_test_data(self):
        self._load_dataset()
        return self._raw_dataset["noise-data"]["test"]

    @property
    def training_trajectory_lengths(self):
        return self._raw_dataset["actual_traj_lens"]["training"]

    @property
    def test_trajectory_lengths(self):
        return self._raw_dataset["actual_traj_lens"]["test"]

    @property
    def default_observation_length(self):
        self._load_dataset()
        return self._raw_dataset["meta"]["default_obs_len"]

    @property
    def min_observation_length(self):
        self._load_dataset()
        return self._raw_dataset["meta"]["min_obs_len"]

    @property
    def max_observation_length(self):
        self._load_dataset()
        return self._raw_dataset["meta"]["max_obs_len"]

    @property
    def max_representation_length(self):
        self._load_dataset()
        return self._raw_dataset["meta"]["max_repr_len"]

    def partitioned_joint_distribution(self, sequence_length=None):
        pi = self._raw_dataset["distribution"]["pi"]
        mu = self._raw_dataset["distribution"]["mu"]
        sigma = np.array([np.diag(m) for m in self._raw_dataset["distribution"]["sigma"]])

        if sequence_length is None:
            sequence_length = len(mu[0]) // 2

        return pi, np.reshape(mu, [-1, sequence_length, 2]), np.array([np.diag(s) for s in np.reshape(sigma, [-1, sequence_length, 2])])

    def partitioned_conditional_joint_distribution(self, observation: np.ndarray):
        assert self.is_gmm_dataset, "Dataset does not provide distribution information"

        pi = self._raw_dataset["distribution"]["pi"]
        mu = self._raw_dataset["distribution"]["mu"]
        sigma = np.array([np.diag(m) for m in self._raw_dataset["distribution"]["sigma"]])

        obs = np.reshape(observation, [-1])

        # partition GMM and calculate conditional GMM parameters -> p(x) -> p(x_obs, x_pred)
        #
        # pi_b|a = pi_k N(x_a|mu_a, E_aa) / sum pi_k N(x_a|mu_a, E_aa)
        # m_b|a = m_b - E_ba * E^-1_aa * (x_a - m_a)
        # E_b|a = E_bb - E_ba * E^-1_aa * E_ab
        # with a = observed portion and b = future portion
        obs_len = observation.shape[-2]
        cond_mu = np.array([mu[k, 2*obs_len:] - np.dot(np.dot(sigma[k, 2*obs_len:, :2*obs_len], np.linalg.inv(sigma[k, :2*obs_len, :2*obs_len])), obs - mu[k, :2*obs_len]) for k in range(len(mu))])
        cond_sigma = np.array([sigma[k, 2*obs_len:, 2*obs_len:] - np.dot(np.dot(sigma[k, 2*obs_len:, :2*obs_len], np.linalg.inv(sigma[k, :2*obs_len, :2*obs_len])), sigma[k, :2*obs_len, 2*obs_len:]) for k in range(len(mu))])
        cond_pi_unnormalized = np.array([pi[k] * stats.multivariate_normal(mu[k, :2*obs_len], sigma[k, :2*obs_len, :2*obs_len]).pdf(obs) for k in range(len(mu))])
        cond_pi = cond_pi_unnormalized / np.sum(cond_pi_unnormalized)

        return cond_pi, np.reshape(cond_mu, [-1, obs_len, 2]), np.array([np.diag(s) for s in np.reshape(cond_sigma, [-1, obs_len, 2])])  # TODO: ich glaube nicht, dass die shapes stimmen...

    def conditional_distribution_at_time(self, observation: np.ndarray, point_in_time: int):
        # targeted point_in_time is measured from the end of the observation: obs_len + t
        assert self.is_gmm_dataset, "Dataset does not provide distribution information"

        pi = self._raw_dataset["distribution"]["pi"]
        mu = self._raw_dataset["distribution"]["mu"]
        sigma = self._raw_dataset["distribution"]["sigma"]

        # partition GMM and calculate conditional GMM parameters -> p(x) -> p(x_obs, x_pred)
        #
        # pi_b|a = pi_k N(x_a|mu_a, E_aa) / sum pi_k N(x_a|mu_a, E_aa)
        # m_b|a = m_b - E_ba * E^-1_aa * (x_a - m_a)
        # E_b|a = E_bb - E_ba * E^-1_aa * E_ab
        # with a = observed portion and b = future portion
        obs_len = observation.shape[-2]
        cond_mu = np.array([mu[k, obs_len:] - np.dot(np.dot(sigma[k, obs_len:, :obs_len], np.linalg.inv(sigma[k, :obs_len, :obs_len])), observation - mu[k, :obs_len]) for k in range(len(mu))])
        cond_sigma = np.array([sigma[k, obs_len:, obs_len:] - np.dot(np.dot(sigma[k, obs_len:, :obs_len], np.linalg.inv(sigma[k, :obs_len, :obs_len])), sigma[k, :obs_len, obs_len:]) for k in range(len(mu))])
        cond_pi_unnormalized = np.array([pi[k] * stats.multivariate_normal(mu[k, :obs_len], sigma[k, :obs_len, :obs_len]).pdf(observation) for k in range(len(mu))])
        cond_pi = cond_pi_unnormalized / np.sum(cond_pi_unnormalized)

        # marginalize specific point in time
        marg_mu = cond_mu[:, 2*point_in_time:2*point_in_time + 2]
        marg_sigma = cond_sigma[:, 2*point_in_time:2*point_in_time + 2, 2*point_in_time:2*point_in_time + 2]

        return cond_pi, marg_mu, marg_sigma

    # TODO: estimate conditional distribution at time? -> nutze dafür gmms + bic gradient für bestimmung d. anzahl komponenten --> https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4



"""
balanced_file_dict = {
    "gt_dist": {
        "component_weights": [0.5, 0.5],
        "mean_vectors": [comp_mean_1.tolist(), comp_mean_2.tolist()],
        "covariance_matrices": [np.diag(cov).tolist(), np.diag(cov).tolist()]
    },
    "samples": {
        "train": train_balanced.tolist(),
        "test": test_balanced.tolist()
    }
}


train_biased = np.random.permutation(np.concatenate([trajs_l[:900], trajs_r[:100]], axis=0))  # concat -> shuffle
test_biased = np.hstack([trajs_l[1000:1100], trajs_r[1000:1100]]).reshape([-1, comp_mean_1.shape[0] // 2, 2])
biased_file_dict = {
    "gt_dist": {
        "component_weights": [0.9, 0.1],
        "mean_vectors": [comp_mean_1.tolist(), comp_mean_2.tolist()],
        "covariance_matrices": [np.diag(cov).tolist(), np.diag(cov).tolist()]
    },
    "samples": {
        "train": train_biased.tolist(),
        "test": test_biased.tolist()
    }
}
io.compress_data({
    "balanced": balanced_file_dict,
    "biased": biased_file_dict
}, "datasets/synthetic/tmaze.gz")
"""


class GMMDataset(object):  # TODO: make this a subclass of dataset
    """
    Joint distribution over all time steps is modeled in a single Gaussian Mixture Model by concatenating all time steps [x_1 y_1 x_2 y_3 ... x_N y_N].
    Given this distribution, the (conditional) ground truth distribution for specific points in time can be calculated by first partitioning the distribution into observed and future portions, then calculating the conditional distribution and finally marginalizing.
    All of these distributions are still Gaussian (mixtures).

    Note: In the synthetic datasets used in this benchmark there are no correlations between dimensions and time steps (we have a diagonal covariance matrix).
    """
    def __init__(self, weights, mean_vectors, cov_matrices):
        self.pi = np.copy(weights)
        self.mu = np.copy(mean_vectors)
        self.sigma = np.copy(cov_matrices)
        if len(self.mu.shape) == 1:
            self.mu = np.reshape(self.mu, [1, -1])
            self.sigma = np.reshape(self.sigma, [1, self.mu.shape[1], self.mu.shape[1]])

    def distribution_at_time(self, observation: np.ndarray, point_in_time: int):
        # point_in_time is measured from the end of the observation: obs_len + t

        # partition GMM and calculate conditional GMM parameters -> p(x) -> p(x_obs, x_pred)
        #
        # pi_b|a = pi_k N(x_a|mu_a, E_aa) / sum pi_k N(x_a|mu_a, E_aa)
        # m_b|a = m_b - E_ba * E^-1_aa * (x_a - m_a)
        # E_b|a = E_bb - E_ba * E^-1_aa * E_ab
        # with a = observed portion and b = future portion
        obs_len = observation.shape[-2]
        cond_mu = np.array([self.mu[k, obs_len:] - np.dot(np.dot(self.sigma[k, obs_len:, :obs_len], np.linalg.inv(self.sigma[k, :obs_len, :obs_len])), observation - self.mu[k, :obs_len]) for k in range(len(self.mu))])
        cond_sigma = np.array([self.sigma[k, obs_len:, obs_len:] - np.dot(np.dot(self.sigma[k, obs_len:, :obs_len], np.linalg.inv(self.sigma[k, :obs_len, :obs_len])), self.sigma[k, :obs_len, obs_len:]) for k in range(len(self.mu))])
        cond_pi_unnormalized = np.array([self.pi[k] * stats.multivariate_normal(self.mu[k, :obs_len], self.sigma[k, :obs_len, :obs_len]).pdf(observation) for k in range(len(self.mu))])
        cond_pi = cond_pi_unnormalized / np.sum(cond_pi_unnormalized)

        # marginalize specific point in time
        marg_mu = cond_mu[:, 2*point_in_time:2*point_in_time + 2]
        marg_sigma = cond_sigma[:, 2*point_in_time:2*point_in_time + 2, 2*point_in_time:2*point_in_time + 2]

        return cond_pi, marg_mu, marg_sigma

        """
        # m_b|a = m_b - E_ba * E^-1_aa * (x_a - m_a)
        # E_b|a = E_bb - E_ba * E^-1_aa * E_ab
        # pi_b|a = pi_k N(x_a|mu_a, E_aa) / sum pi_k N(x_a|mu_a, E_aa)
        cond_m1 = lambda x: m1[obs_len:] - np.dot(np.dot(cov[obs_len:, :obs_len], np.linalg.inv(cov[:obs_len, :obs_len])), x - m1[:obs_len])
        cond_m2 = lambda x: m2[obs_len:] - np.dot(np.dot(cov[obs_len:, :obs_len], np.linalg.inv(cov[:obs_len, :obs_len])), x - m2[:obs_len])
        cond_cov = cov[obs_len:, obs_len:] - np.dot(np.dot(cov[obs_len:, :obs_len], np.linalg.inv(cov[:obs_len, :obs_len])), cov[:obs_len, obs_len:])
        cond_pi1 = lambda x: pi1 * stats.multivariate_normal(m1[:obs_len], cov[:obs_len, :obs_len]).pdf(x) / (
                    pi1 * stats.multivariate_normal(m1[:obs_len], cov[:obs_len, :obs_len]).pdf(x) + pi2 * stats.multivariate_normal(m2[:obs_len], cov[:obs_len, :obs_len]).pdf(x))
        cond_pi2 = lambda x: pi2 * stats.multivariate_normal(m2[:obs_len], cov[:obs_len, :obs_len]).pdf(x) / (
                    pi1 * stats.multivariate_normal(m1[:obs_len], cov[:obs_len, :obs_len]).pdf(x) + pi2 * stats.multivariate_normal(m2[:obs_len], cov[:obs_len, :obs_len]).pdf(x))

        cp1 = cond_pi1(sample[:obs_len])
        cp2 = cond_pi2(sample[:obs_len])
        cm1 = cond_m1(sample[:obs_len])
        cm2 = cond_m2(sample[:obs_len])
        print(cp1, cp2)
        for _ in range(100):
            k = np.random.choice(2, p=[cp1, cp2])
            s = np.reshape(np.random.multivariate_normal([cm1, cm2][k], cond_cov), [-1, 2])
            plt.plot(s[-1:, 0], s[-1:, 1], "go")
        """

    # TODO: plotting stuff