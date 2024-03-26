import os

import numpy as np
import matplotlib.pyplot as plt

import stsc.datagen.predefined_datasets as pd
from stsc.datagen.bezier_spline import BezierSpline


def path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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


if __name__ == "__main__":
    base_dir = path("./out/dataset_gen_paper")

    # plot stuff
    gen_figures()
    quit()

    # exemplary dataset
    syn_hyang = pd.synth_hyang()
    syn_hyang.plot_samples(show=True)

    # training

    # other stuff...
