from typing import List
from enum import Enum
import multiprocessing as mp

import numpy as np
import scipy.stats as stats
from ot.sliced import sliced_wasserstein_distance

from stsc.datagen.sequence_sample_set import SequenceSampleSet
from stsc.datagen.gmm_sequence import GMMSequence


class Metrics(Enum):
    ADE = 0
    NLL = 1
    KL_DIVERGENCE = 2
    WASSERSTEIN_DISTANCE = 3
    RESP_COLLAPSE = 4
    RESP_IND_BIC = 5
    RESP_IND_INTRA_DIST = 6
    RESP_IND_INTER_DIST = 7
    RESP_IND_PRED_GT_DIST = 8
    RESP_IND_GM_WEIGHTS = 9


def nll(
        pred_samples: SequenceSampleSet, 
        test_data_pred: np.ndarray, 
        ret_mean: bool = True
    ) -> float:
    """ Calculates the negative log-likelihood given a sample-based prediction and a set of test trajectories. """
    nll = []
    for i in range(pred_samples.n_sequences):
        seq_nll = []
        for step in range(pred_samples.sequence_length):
            kde = pred_samples.step_kde(i, step)
            seq_nll.append((-1) * kde.logpdf(test_data_pred[i, step]))
        nll.append(np.mean(seq_nll))

    if ret_mean:
        return np.mean(nll)
    return nll


def kl_div2(
        pred_samples: SequenceSampleSet,
        gt_gmm_seqs: List[GMMSequence],
        n_gt_samples: int = 500,
        ret_mean: bool = True
    ) -> float:
    # https://ieeexplore.ieee.org/document/4218101
    # D(f||g) = int f(x) log f(x)/g(x) dx, where f(x) is the ground truth distribution and g(x) the predicted distribution
    # For GMM: MC approach: D_MC(f||g) = 1/n sum^n_{i=1} log f(x_i) / g(x_i) -> D(f||g) for n -> infty
    # -> samples x_i are drawn from f. The variance of the estimation error is 1/n Var_f[log f/g]
    kl_div = []
    for i in range(pred_samples.n_sequences):
        seq_kl = []
        for step in range(len(pred_samples.index_kdes[i])):
            gt_samples = gt_gmm_seqs[i](step).sample(n_samples=n_gt_samples)
            seq_kl.append((1. / n_gt_samples) * np.sum([gt_gmm_seqs(step).pdf(sample) / pred_samples.index_kdes[i][step].pdf(sample) for sample in gt_samples]))
        kl_div.append(np.mean(seq_kl))
    if ret_mean:
        return np.mean(kl_div)
    return kl_div


def kl_div(
        pred_samples: SequenceSampleSet, 
        gt_gmm_seqs: List[GMMSequence], 
        test_data_obs: np.ndarray, 
        n_processes: int = -1, 
        ret_mean: bool = True
    ) -> float:
    """
    Calculates the KL divergence between sample-based predicted sequence distribution and the ground truth distribution.
    Assumes that observed segment starts at the first point of each path in the TrajectoryGMM. 

    This operation is parallelized over the test sequences (trajectories), as there is no closed-form solution to the KL divergence, thus the computation involves sampling from the ground truth distribution.
    """    
    if n_processes == 1:
        kl_div = []
        for i in range(len(test_data_obs)):
            kl_div.append(_seq_kl(pred_samples.index_samples(i), pred_samples.index_kdes(i), gt_gmm_seqs[i]))
    else:
        if n_processes == -1:
            n_processes = np.maximum(1, np.minimum(mp.cpu_count() - 1, 8))
        with mp.Pool(mp.cpu_count() - 1) as p:
            kl_div = p.starmap(_seq_kl, [(pred_samples.index_samples(i), pred_samples.index_kdes(i), gt_gmm_seqs[i]) for i in range(len(test_data_obs))])
            p.terminate()  # TODO: not explicitly calling terminate() sometimes seems to interfere with pyplot 
    
    if ret_mean:
        return np.mean(kl_div)
    return kl_div


def _seq_kl(
        pred_step_samples: np.ndarray, 
        pred_step_kdes: List[stats.gaussian_kde], 
        gt_gmm_seq: GMMSequence, 
        n_gt_samples: int = 100
    ) -> float:
    seq_kl = []
    for step in range(len(pred_step_kdes)):  
        gt_samples = gt_gmm_seq(step).sample(n_samples=n_gt_samples)
        # in theory, infinitely many points need to be sampled for calculating an accurate pdf, but all points which are far enough away from both given probability distributions yield approx. 0.
        # -> use samples of both probability distributions as spatial positions in order to calculate the KL divergence
        test_samples = np.concatenate([gt_samples, pred_step_samples[step]], axis=0)

        gt_pdf_val = np.maximum(gt_gmm_seq(step).pdf(test_samples), 1e-6)
        pred_pdf_val = np.maximum(pred_step_kdes[step].pdf(np.swapaxes(test_samples, 0, 1)), 1e-6)
        seq_kl.append( np.mean( gt_pdf_val * np.log(gt_pdf_val / pred_pdf_val) ) )
    return np.mean(seq_kl)


def seq_gauss_kl_div(gt_sseq, p_sseq, zero_mean=False):
    """
    Estimates Gaussian distributions from sample sets (should only be used if a single Gaussian is to be expected for data at hand).
    This is simply sample mean and sample covariance.
    Then calculates (closed-form) KL divergence for these Gaussians.
    """
    kl_vals = []
    for ex_i in range(p_sseq.n_sequences):            
        kl_seq = []
        for step in range(p_sseq.sequence_length):
            pmu, pcov = p_sseq.step_mean_cov(ex_i, step)
            gtmu, gtcov = gt_sseq.step_mean_cov(ex_i, step)
            # use zero mean for both distributions if we are only interested in comparing the variances
            if zero_mean:
                kl_seq.append(_gauss_kl_div(np.zeros_like(gtmu), gtcov, np.zeros_like(pmu), pcov))
            else:
                kl_seq.append(_gauss_kl_div(gtmu, gtcov, pmu, pcov))
        kl_vals.append(np.mean(kl_seq))
    return kl_vals


def _gauss_kl_div(mu_gt, cov_gt, mu_pred, cov_pred):
    # D(P_gt||P_pred), with P_gt ~ N(mu_gt, cov_gt), P_pred ~ N(mu_pred, cov_pred) and dimension d
    # Here, P_gt is the true/gt distribution and P_pred is the approximation given by a model.
    # https://stanford.edu/~jduchi/projects/general_notes.pdf
    d = len(mu_gt)
    mu_gt = mu_gt.reshape([-1, 1])
    mu_pred = mu_pred.reshape([-1, 1])

    log = np.log(np.linalg.det(cov_pred) / np.linalg.det(cov_gt))
    trace = np.trace(np.linalg.inv(cov_pred) @ cov_gt)
    mahalanobis = ((mu_pred  - mu_gt).T @ np.linalg.inv(cov_pred)) @ (mu_pred - mu_gt)
    return 0.5 * (log - d + trace + mahalanobis)


def wasserstein(
        pred_samples: SequenceSampleSet, 
        gt_gmm_seqs: List[GMMSequence], 
        test_data_obs: np.ndarray, 
        n_gt_samples: int = 100, 
        n_seeds: int = 10, 
        n_projections: int = 200, 
        ret_mean: bool = True
    ) -> float:
    """
    Calculates the Wasserstein distance between a sample-based predicted sequence distribution and the ground truth distribution.
    Applies the sample-based sliced wasserstein distance as an approximate of the true Wasserstein distance.
    Assumes that observed segment starts at the first point of each path in the TrajectoryGMM. 
    """
    res = []
    for i in range(pred_samples.n_sequences):
        gmm_seq = gt_gmm_seqs[i]

        wseq = []
        for step in range(pred_samples.sequence_length):
            p_samples = pred_samples.step_samples(i, step)
            gt_samples = gmm_seq(step).sample(n_samples=n_gt_samples)
            res_step = np.empty(n_seeds)
            for seed in range(n_seeds):
                res_step[seed] = sliced_wasserstein_distance(p_samples, gt_samples, n_projections=n_projections, seed=seed)
            wseq.append(np.mean(res_step))
        res.append(np.mean(wseq))

    if ret_mean:
        return np.mean(res)
    return res


def ade(pred_samples: SequenceSampleSet, test_data_pred: np.ndarray, ret_mean: bool = True) -> float:
    """ Calculates the average displacement error by comparing a maximum likelihood prediction computed from the sample-based predicted sequence distribution with a set of test trajectories. """
    dists = []
    for i in range(pred_samples.n_sequences):
        seq_dists = []
        for step in range(pred_samples.sequence_length):
            step_mean = np.mean(pred_samples.step_samples(i, step), axis=0)
            seq_dists.append(np.linalg.norm(step_mean - test_data_pred[i, step]))
        dists.append(np.mean(seq_dists))
    
    if ret_mean:
        return np.mean(dists)
    return dists
