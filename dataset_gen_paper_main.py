import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from stsc.datagen.bezier_spline import BezierSplineBuilder2D
from stsc.datagen.bezier_spline import BezierSplineBuilder2D as BSB
from stsc.datagen.bezier_spline import BezierSpline
from stsc.datagen.prob_bezier_spline import ProbabilisticBezierSpline
from stsc.datagen.trajectory_gmm import TrajectoryGMM
import stsc.benchmark.evaluation.metrics as metr
from stsc.datagen.sequence_sample_set import SequenceSampleSet
from stsc.datagen.common import confidence_ellipse
from main_script_utils.common.io import path
from main_script_utils.common.colors import STSCColors
from main_script_utils.dataset_gen_paper.dataset import gen_dataset
from main_script_utils.dataset_gen_paper.models import get_models
from main_script_utils.dataset_gen_paper.plots import gen_figures


if __name__ == "__main__":
    base_dir = path("./out/dataset_gen_paper")
    obs_len = 4
    pred_len = 6

    # exemplary dataset
    dataset_file = os.path.join(base_dir, "dataset.pkl")
    if not os.path.exists(dataset_file):
        sample_dataset = gen_dataset()
        sample_dataset.save(dataset_file)
    else:
        sample_dataset = TrajectoryGMM.from_file(dataset_file)

    # TODO: plot mal noch sample von comp 2 dazu um zu schauen ob sie aligned sind (mit start 6 ) + dataset marginal ab index 6
        
    # prepare training and test datasets
    data_rng = np.random.default_rng(1)
    training_data = np.asarray(sample_dataset.sample(200, cap_length=19, rng=data_rng))
    test_data_full = sample_dataset.sample(20, cap_length=19, rng=data_rng)
    test_data = []
    for traj in test_data_full:
        for i in range(len(traj) - (obs_len + pred_len) + 1):
            test_data.append(traj[i:i+obs_len+pred_len])
    test_data = np.asarray(test_data)

    # calculate ground truth posterior distributions for all test samples
    #posteriors_file = os.path.join(base_dir, "posteriors.pkl")
    #if not os.path.exists()
    test_posteriors = []
    test_posterior_samples = []
    for traj in test_data:
        sample_dataset.posterior(traj[:obs_len])
        test_posteriors.append(sample_dataset.sequence_distribution())
    sample_dataset.prior()  # reset dataset back to prior state
    
    # models
    torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mm_red, vae, gan = get_models(obs_len, pred_len, torch_device)

    # model training
    red_ckpt = os.path.join(base_dir, "red")
    if not os.path.exists(f"{red_ckpt}.pth"):
        mm_red.estimate_parameters(training_data, 50, obs_len, pred_len, n_epochs=50, verbose=True)
        mm_red.save(red_ckpt)
    else:
        mm_red.load_state_dict(torch.load(f"{red_ckpt}.pth", map_location=torch_device))
    
    vae_ckpt = os.path.join(base_dir, "vae")
    if not os.path.exists(f"{vae_ckpt}.pth"):
        vae.estimate_parameters(training_data, 50, obs_len, pred_len, n_epochs=50, verbose=True) 
        vae.save(vae_ckpt)
    else:
        vae.load_state_dict(torch.load(f"{vae_ckpt}.pth", map_location=torch_device)) 

    gan_ckpt = os.path.join(base_dir, "gan")
    if not os.path.exists(f"{gan_ckpt}.pth"):
        gan.estimate_parameters(training_data, 50, obs_len, pred_len, n_epochs=50, verbose=True)
        gan.save(gan_ckpt)
    else:
        gan.load_state_dict(torch.load(f"{gan_ckpt}.pth", map_location=torch_device)) 

    # model evaluation
    # pred target shape: [ n_sequences, sequence_length, n_samples, 2 ]
    """red_pred = SequenceSampleSet(mm_red.predict(test_data[:, :obs_len], sample_pred=True))
    print("MM_RED")
    print("NLL:", metr.nll(red_pred, test_data[:, obs_len:], ret_mean=True))
    print("KL:", metr.kl_div(red_pred, test_posteriors, test_data[:, :obs_len], ret_mean=True))
    print("KL2:", metr.kl_div2(red_pred, test_posteriors))
    print("EMD:", metr.wasserstein(red_pred, test_posteriors, test_data[:, :obs_len], ret_mean=True, n_seeds=5, n_projections=100))
    print()

    vae_pred = SequenceSampleSet(np.swapaxes(vae.predict(test_data[:, :obs_len], pred_len, num_samples=50), 1, 2))  # vae pred shape: [observation.shape[0], num_samples, num_steps, 2]
    print("VAE")
    print("NLL:", metr.nll(vae_pred, test_data[:, obs_len:], ret_mean=True))
    print("KL:", metr.kl_div(vae_pred, test_posteriors, test_data[:, :obs_len], ret_mean=True))
    print("EMD:", metr.wasserstein(vae_pred, test_posteriors, test_data[:, :obs_len], ret_mean=True, n_seeds=5, n_projections=100))
    print()

    gan_pred = SequenceSampleSet(np.swapaxes(gan.predict(test_data[:, :obs_len], pred_len, num_samples=50), 1, 2))  # gan pred shape: [observation.shape[0], num_samples, num_steps, 2]
    print("GAN")
    print("NLL:", metr.nll(gan_pred, test_data[:, obs_len:], ret_mean=True))
    print("KL:", metr.kl_div(gan_pred, test_posteriors, test_data[:, :obs_len], ret_mean=True))
    print("EMD:", metr.wasserstein(gan_pred, test_posteriors, test_data[:, :obs_len], ret_mean=True, n_seeds=5, n_projections=100))
    print()"""

    # other stuff...    
        
    # plot stuff
    gen_figures(base_dir, sample_dataset, obs_len)

