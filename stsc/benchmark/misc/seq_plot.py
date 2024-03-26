import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plot_gaussian_sequence_mixture(weights, means, covs, suppress_alpha=False, axis=None, show=True):
    """ 
    Plot mixture of sequences comprised of Gaussians. 
    
    For clarity: Each mixture component consists of a sequence of Gaussians.
    This is opposed to <plot_gmm_sequence>, where each element in the given sequence is itself a mixture of Gaussians.
    """
    if axis is None:
        axis = plt.gca()

    n_comps = len(weights)
    for comp in range(n_comps):
        if weights[comp] < 1e-6:
            continue

        mseq = means[comp]
        alpha = 1. if suppress_alpha else weights[comp]

        lines = axis.plot(mseq[:, 0], mseq[:, 1], "o-", alpha=alpha, label=f"{weights[comp]:.2f}")
        cur_color = lines[0].get_color()
        axis.plot(mseq[:1, 0], mseq[:1, 1], "o", color=cur_color, alpha=alpha, markerfacecolor="none", markersize=10)

        for t in range(len(mseq)):
            step_cov = covs[comp][t]
            confidence_ellipse(mseq[t], step_cov, axis, 3, cur_color, "none", alpha*0.33)
            confidence_ellipse(mseq[t], step_cov, axis, 2, cur_color, "none", alpha*0.66)
            confidence_ellipse(mseq[t], step_cov, axis, 1, cur_color, "none", alpha*1.)

    plt.legend()
    if show:
        plt.show()


def plot_gmm_sequence():
    """ 
    Plot sequence of Gaussian mixtures. 

    For clarity: Each element in the given sequence is a mixture of Gaussians.
    This is opposed to <plot_gaussian_sequence_mixture>, where each mixture component consists of a sequence of Gaussians.
    """
    ...


def plot_discrete_density_sequence(draw_kde: bool = False):
    ...


def confidence_ellipse(mean, cov, axis, n_std, edgecolor, facecolor, alpha):
    # https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    # https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]))

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + axis.transData)
    axis.add_patch(ellipse)
