import numpy as np

from stsc.datagen.bezier_spline import BezierSplineBuilder2D
from stsc.datagen.bezier_spline import BezierSplineBuilder2D as BSB
from stsc.datagen.prob_bezier_spline import ProbabilisticBezierSpline
from stsc.datagen.trajectory_gmm import TrajectoryGMM


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