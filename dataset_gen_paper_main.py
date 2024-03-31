import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from stsc.datagen.bezier_spline import BezierSplineBuilder2D
from stsc.datagen.bezier_spline import BezierSplineBuilder2D as BSB
from stsc.datagen.bezier_spline import BezierSpline
from stsc.datagen.prob_bezier_spline import ProbabilisticBezierSpline
from stsc.datagen.trajectory_gmm import TrajectoryGMM
from stsc.benchmark.baseline_models.red import REDPredictor2D
from stsc.benchmark.baseline_models.vae import VAESeqPredictor2D
from stsc.benchmark.baseline_models.gan import GANSequencePredictor2D
import stsc.benchmark.evaluation.metrics as metr
from stsc.datagen.sequence_sample_set import SequenceSampleSet


def path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def gen_dataset():
    """
    Reduced Walking-Path Structure imitating sdd-hyang:

    ######## 2 2 2 ########
    ########       ########
    ########       ########
    1                     3
    1 (0,0)               3
    1                     3
    ########       ########
    ########              4
    ########              4 (8,-5)
    ########              4
    ########       ########
    ######## 5 5 5 ########
             (4,-7)

    Origin is located in source/sink "1"

    Paths:
    4 -> 1 [1 straight, 2x2 curve, 1 straight, 2x2 curve, 1 straight]
    4 -> 2 [1 straight, 2x2 curve, 1+2+1 straight]
    5 -> 3 [1+2+1 straight, 2x2 curve, 1 straight]
    ------
    = 6 paths, i.e. mixture components
    """
    paths = [      
        # 4 -> 1
        BezierSplineBuilder2D(origin=np.array([6., -6]), initial_dir=np.array([-1., 0.])).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(2)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(2)).instantiate_spline(),
        # 4 -> 2
        BezierSplineBuilder2D(origin=np.array([6., -6]), initial_dir=np.array([-1., 0.])).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(8)).instantiate_spline(),
        # 5 -> 3
        BezierSplineBuilder2D(origin=np.array([4., -7.75]), initial_dir=np.array([0., 1.])).add(BSB.LineSegment(2+2+0.25)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(2)).instantiate_spline()
    ]

    weights = [1 / len(paths) for _ in range(len(paths))]
    means = []
    covs = []
    target_variance = 0.075**2 
    point_distance = 0.5
    for path in paths:
        pspline = ProbabilisticBezierSpline.approximate_uniform_variance(path, target_variance, elevate_segments_degree=10)
        m, c = pspline.gp_discretize(point_distance=point_distance) 
        means.append(m)
        covs.append(c)

    return TrajectoryGMM(
        weights=weights,
        means=means,
        covs=covs,
        name="sample_dataset"
    )


def get_models(device: str = "cuda:0"):
    return (
        REDPredictor2D(obs_len, 64, 1, pred_len, 5, device),
        VAESeqPredictor2D(offset_mode=True, torch_device=device),
        GANSequencePredictor2D(torch_device=device)
    )


def gen_figures():
    _spline_continuity()


def _spline_continuity():
    seg1 = np.reshape([0, 0, 1, 1, 2, 0], [3, 2])
    seg2 = np.reshape([2, 0, 2.5, -0.5, 4, 0], [3, 2])
    seg3 = np.reshape([2, 0, 2, -1, 3, -2], [3, 2])
    b1 = BezierSpline([seg1, seg2])
    b2 = BezierSpline([seg1, seg3])

    b1.plot(connect_control_pts=True, highlight_start=True, show=False)
    plt.savefig(f"{base_dir}/c1_spline.png", dpi=300, bbox_inches="tight")
    plt.close()
    b2.plot(connect_control_pts=True, highlight_start=True, show=False)
    plt.savefig(f"{base_dir}/c0_spline.png", dpi=300, bbox_inches="tight")
    plt.close()


def _spline_curve_pts():
    ... # 3 segmente, 1 punkt pro segment zeichnen und verbindungen zu den jeweiligen kontrollpunkten


def _spline_curve_pt_kernel_pairs():
    ... # grafik die die 3 fälle für covariance function zeigt. kernel is matrix-valued zeige neben grafik auch resultierenden matrixblock


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
    mm_red, vae, gan = get_models(torch_device)

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
    #gen_figures()
    sample_dataset.plot(show=False)
    #sample_dataset.plot_samples(show=True)
    sample_traj = sample_dataset.sample_component(0, 1, rng=np.random.default_rng(123), scale_variance=0.5)[0] 
    plt.plot(sample_traj[:, 0], sample_traj[:, 1], "kx--")
    plt.show()

    for start in [0, 6, 12]:
        sample_dataset.posterior(sample_traj[start:start+obs_len])
        plt.figure()
        plt.plot(sample_traj[:, 0], sample_traj[:, 1], "k--")
        #plt.plot(sample_traj[start:start+obs_len, 0], sample_traj[start:start+obs_len, 1], "ko") 
        sample_dataset.plot(show=False, suppress_alpha=True)
        plt.xlim([0, 10])
        plt.ylim([-10, 5])
        plt.show()
