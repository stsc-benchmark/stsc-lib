import numpy as np
import matplotlib.pyplot as plt

from stsc.datagen.bezier_spline import BezierSpline
from stsc.datagen.prob_bezier_spline import ProbabilisticBezierSpline
from stsc.datagen.trajectory_gmm import TrajectoryGMM
from stsc.datagen.common import confidence_ellipse
from main_script_utils.common.colors import STSCColors


def gen_figures(base_dir: str, traj_gmm: TrajectoryGMM, obs_len):
    #_spline_continuity(base_dir)
    _prob_spline(base_dir)
    _ngp_prior(base_dir)
    _posterior_example(base_dir, traj_gmm, obs_len)
    _eval_dataset(base_dir, traj_gmm)


def _spline_continuity(base_dir: str):
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


def _prob_spline(base_dir: str):
    s1 = np.reshape([0, 0, 1, 2, 2, 0], [3, 2])
    s2 = np.reshape([2, 0, 2.5, -1, 3, 1.75, 4, 0], [4, 2])
    s3 = np.reshape([4, 0, 4.5, -0.875, 3.5, -1.75], [3, 2])
    b = BezierSpline([s1, s2, s3])
    v1, v2 = 0.1**2, 0.3**2
    pb = ProbabilisticBezierSpline(b, [v1 * np.ones([3, 2]), np.reshape([v1, v1, v2, v1, v2, v1, v2, v1], [4, 2]), np.reshape([v2, v1, v1, v2, v1, v2], [3, 2])])
    
    colors = [STSCColors.Red.value, STSCColors.Green.value, STSCColors.Blue.value]
    b.plot(connect_control_pts=True, 
           color_code_segments=True,
           segment_colors=colors,
           highlight_start="all",
           show=False)
    plt.xlim([-0.25, 5])
    plt.ylim([-2.5, 2.25])
    plt.savefig(f"{base_dir}/prob_spline_1.png", dpi=300, bbox_inches="tight")
    plt.close()

    b.plot(connect_control_pts=True, 
           color_code_segments=False, 
           line_style=":", 
           highlight_start="all",
           show=False)    
    for i, t in enumerate([0.15, 0.55, 0.8]):
        seg_i, seg_t = b.map_t(t)
        pt = b.curve_point(t)
        _, pt_cov = pb.gaussian_bezier_curve_point(seg_t, b.segments[seg_i], pb.segment_covariance_matrices[seg_i])

        plt.plot([pt[0]], [pt[1]], "o", color=colors[i])
        confidence_ellipse(pt, pt_cov, plt.gca(), 2, colors[i], "none", 0.66)
        confidence_ellipse(pt, pt_cov, plt.gca(), 1, colors[i], "none", 1.)
        for j, cpt in enumerate(b.segments[seg_i]):
            plt.plot([pt[0], cpt[0]], [pt[1], cpt[1]], "-", color=colors[i])
            plt.plot([cpt[0]], [cpt[1]], "ko")
            cpt_cov = pb.segment_covariance_matrices[seg_i][j]
            confidence_ellipse(cpt, cpt_cov, plt.gca(), 2, "k", "none", 0.66, "--")
            confidence_ellipse(cpt, cpt_cov, plt.gca(), 1, "k", "none", 1., "--") 
    plt.xlim([-0.25, 5])
    plt.ylim([-2.5, 2.25])           
    plt.savefig(f"{base_dir}/prob_spline_2.png", dpi=300, bbox_inches="tight")
    plt.close()


def _ngp_prior(base_dir):
    s1 = 10*np.reshape([0, 0, 1, 2, 2, 0], [3, 2])
    s2 = 10*np.reshape([2, 0, 2.5, -1, 3, 1.75, 4, 0], [4, 2])
    s3 = 10*np.reshape([4, 0, 4.5, -0.875, 3.5, -1.75], [3, 2])
    s4 = 10*np.reshape([3.5, -1.75, 2.5, -2.625, 1, -1.75], [3, 2])
    b = BezierSpline([s1, s2, s3, s4])
    v = (20 * 0.1)**2
    pb = ProbabilisticBezierSpline(b, [v * np.ones([3, 2]), v * np.ones([4, 2]), v * np.ones([3, 2]), v * np.ones([3, 2])])
    t_vals = np.array([0.1, 0.2, 0.4, 0.9])
    pts = np.array([b.curve_point(t) for t in t_vals])

    gp_mean, gp_cov = pb.gp_discretize(fixed_curve_pts=t_vals)
    print(gp_mean)
    print(gp_cov)

    b.plot(connect_control_pts=True, color_code_segments=False, highlight_start="none", show=False)
    for i in range(len(b.segments)):
        for j, cpt in enumerate(b.segments[i]):
            cpt_cov = pb.segment_covariance_matrices[i][j]
            confidence_ellipse(cpt, cpt_cov, plt.gca(), 2, "k", "none", 0.66, "--")
            confidence_ellipse(cpt, cpt_cov, plt.gca(), 1, "k", "none", 1., "--") 
    for i, pt in enumerate(pts):
        plt.plot(*pt, "s", color=[STSCColors.Red.value, STSCColors.Green.value, STSCColors.Blue.value, STSCColors.Magenta.value][i], markersize=7.5)
    plt.savefig(f"{base_dir}/gp_discretize.png", dpi=300, bbox_inches="tight")
    plt.close()


def _posterior_example(base_dir: str, traj_gmm: TrajectoryGMM, obs_len: int):
    traj_gmm.plot(show=False)
    #sample_dataset.plot_samples(show=True)
    sample_traj = traj_gmm.sample_component(0, 1, rng=np.random.default_rng(123), scale_variance=0.5)[0] 
    plt.plot(sample_traj[:, 0], sample_traj[:, 1], "kx--")
    plt.xlim([0, 10])
    plt.ylim([-10, 5])
    plt.savefig(f"{base_dir}/dataset_prior.png", dpi=300, bbox_inches="tight")
    plt.close()

    for i, start in enumerate([0, 6, 12]):
        traj_gmm.posterior(sample_traj[start:start+obs_len])
        plt.figure()
        plt.plot(sample_traj[:, 0], sample_traj[:, 1], "k--")
        #plt.plot(sample_traj[start:start+obs_len, 0], sample_traj[start:start+obs_len, 1], "ko") 
        traj_gmm.plot(show=False, suppress_alpha=True)
        plt.xlim([0, 10])
        plt.ylim([-10, 5])
        plt.savefig(f"{base_dir}/dataset_posterior_{i}.png", dpi=300, bbox_inches="tight")
        plt.close() 


def _eval_dataset(base_dir: str, traj_gmm: TrajectoryGMM):
    traj_gmm.plot()
    plt.savefig(f"{base_dir}/train_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()

    samples = traj_gmm.plot_samples(num_samples=5)
    plt.savefig(f"{base_dir}/train_dataset_samples.png", dpi=300, bbox_inches="tight")
    plt.close()

