# setup script for initializing all benchmark files (generates data sets and ground truth data)
# evaluation factor init functions use a fixed seed to ensure reproducible results.

import numpy as np

from stsc.benchmark.evaluation.factors.outlier_robustness import OutlierRobustness
from stsc.benchmark.evaluation.factors.overall_performance import OverallPerformance
from stsc.benchmark.evaluation.factors.responsiveness import Responsiveness
from stsc.benchmark.evaluation.factors.variance_estimation import VarianceEstimation


print("Initializing evaluation factor files...")

print("Running <OutlierRobustness> init")
OutlierRobustness(np.random.default_rng(1))

print("Running <OverallPerformance> init")
OverallPerformance(np.random.default_rng(2))

print("Running <Responsiveness> init")
Responsiveness(np.random.default_rng(3))

print("Running <VarianceEstimation> init")
VarianceEstimation(np.random.default_rng(4))
