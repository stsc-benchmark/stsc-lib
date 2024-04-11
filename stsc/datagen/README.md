This package contains an implementation of a ground truth generation approach for generating probability distributions over full trajectories based on (mixtures of) probabilistic composite Bézier curves as described in our [short paper](https://arxiv.org/abs/2404.04397). Note that composite Bézier curves are oftentimes also referred to as Bézier splines. Both terms are used interchangeably in this repository.

In order to define a simple dataset covering multiple paths through a virtual scene with approximately equal variance for each trajectory point, a few steps are required:

**Path definition**:
Define paths in terms of a list of [BezierSpline](bezier_spline.py) instances. For convenience, in this module we additionally provide the `BezierSplineBuilder2D` class, which allows the definition of a spline by stringing together segment primitives (lines and curves). For example:
```python
from stsc.datagen.bezier_spline import BezierSplineBuilder2D
from stsc.datagen.bezier_spline import BezierSplineBuilder2D as BSB

path = BezierSplineBuilder2D(origin=np.array([5., 5.]), initial_dir=np.array([0., -1.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).instantiate_spline()
```
More examples can be found in [predefined_datasets.py](predefined_datasets.py).

**Add uncertainty**:
Next we have to instantiate a [ProbabilisticBezierSpline](prob_bezier_spline.py) from each path Bezier spline. For targeting uniform variance along in each, `ProbabilisticBezierSpline.approximate_uniform_variance` can be used, which assigns diagonal covariance matrices to each spline control point and optimizes each variance towards a uniform variance close to the given target along the spline curve. Afterwards, the `ProbabilisticBezierSpline` is discretized to a sequence of `N` Gaussians using `ProbabilisticBezierSpline.gp_discretize`, extracts a probability distribution over `N` trajectory points along the defined curve using a Gaussian process representation of the curve together with some *policy* for the placement of points along the curve (see below). This yields a flattened mean vector for all Gaussian trajectory point means and a block-partitioned covariance matrix covering all `N` steps and their correlations. Regarding the placement policies, it is important to note, that the position on a spline curve is determined by a position parameter $t \in [0, 1]$, thus a policy dictates how a set of `N` values of `t` is put together for the discretization. Available policies include 
- Equidistant Positions: Spreads the values for `t` equally across the interval `[0, 1]`. The relative distance between resulting curve points then depends on the positions of the control points.
- Equidistant Points: Chooses the values for `t`, such that the euclidean distance between subsequent curve points is equal.
- Fixed positions: The user choses the values for `t`.
- Acceleration: Chooses the values for `t`, such that the euclidean distance between subsequent curve points gradually increases or decreases.

**Instantiate the Dataset**:
The trajectory dataset based around distributions over full trajectories is handled by the [TrajectoryGMM](trajectory_gmm.py) class and is instantiated using the list of resulting mean vectors and covariance matrices obtained from `ProbabilisticBezierSpline.gp_discretize` (path weights have to be defined as a final input). In this Gaussian mixture, each component covers one path with a fixed velocity profile. The `TrajectoryGMM` class is built as a state machine provides functions for switching its state, sampling and plotting. Available states are *prior* and *posterior*. Given some trajectory observation, the dataset can be switched into posterior mode, providing the posterior distribution calculated by conditioning the prior distribution on the observation. Important: When conditioning on some trajectory, for each mixture component in the dataset, the best matching subsequence is assigned to the observation and the posterior distribution is only calculated for time steps *after* the observation. For example, the following code

```python
from main_script_utils.dataset_gen_paper.dataset import gen_dataset

# dataset starts in prior mode
dataset = gen_dataset()  

# sample 1 trajectory from 0'th component. 
# the component covers a path with 21 points 
sample_traj = dataset.sample_component(0, 1)  

# switch dataset into posterior mode by conditioning on a sub-trajectory (observation)
dataset.posterior(sample_traj[:4])  
```

yields the prior and posterior as displayed here:

<p float="middle">
  <img src="../../out/dataset_gen_paper/dataset_prior.png" width="300" />
  <img src="../../out/dataset_gen_paper/dataset_posterior_0.png" width="300" /> 
</p>

Note that for visualization purposes, each respective trajectory distribution is displayed as a sequence of marginal distributions for each trajectory point. The trajectory and observation is depicted in black.
