import numpy as np

from stsc.datagen.bezier_spline import BezierSplineBuilder2D
from stsc.datagen.bezier_spline import BezierSplineBuilder2D as BSB
from stsc.datagen.prob_bezier_spline import ProbabilisticBezierSpline
from stsc.datagen.trajectory_gmm import TrajectoryGMM


def balanced_tmaze():
    left_spline = BezierSplineBuilder2D().add(BSB.LineSegment(3)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(2)).instantiate_spline()
    left_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(left_spline, 0.005, elevate_segments_degree=10)
    l_mean, l_cov = left_pspline.gp_discretize(point_distance=0.5)

    right_spline = BezierSplineBuilder2D().add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(2)).instantiate_spline()
    right_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(right_spline, 0.005, elevate_segments_degree=10)
    r_mean, r_cov = right_pspline.gp_discretize(point_distance=0.5)

    return TrajectoryGMM(
        weights=[0.5, 0.5],
        means=[l_mean, r_mean],
        covs=[l_cov, r_cov],
        sampling_seed=123,
        name="tmaze"
    )


def balanced_tmaze_longtail():
    left_spline = BezierSplineBuilder2D().add(BSB.LineSegment(2)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline()
    left_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(left_spline, 0.005, elevate_segments_degree=10)
    l_mean, l_cov = left_pspline.gp_discretize(point_distance=0.5)

    right_spline = BezierSplineBuilder2D().add(BSB.LineSegment(2)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline()
    right_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(right_spline, 0.005, elevate_segments_degree=10)
    r_mean, r_cov = right_pspline.gp_discretize(point_distance=0.5)

    return TrajectoryGMM(
        weights=[0.5, 0.5],
        means=[l_mean, r_mean],
        covs=[l_cov, r_cov],
        sampling_seed=123,
        name="tmaze_longtail"
    )


def diamond():
    sl = 10 
    var = 0.01
    cap_len = 8 
    norm_factor = np.linalg.norm([1., 1.])
    
    slow_spline = BezierSplineBuilder2D(initial_dir=np.array([1., 1.]) / norm_factor).add(BSB.LineSegment(sl)).instantiate_spline()
    slow_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(slow_spline, var, elevate_segments_degree=10)
    s_mean, s_cov = slow_pspline.gp_discretize(point_distance=0.5, sequence_length_cap=cap_len)
    
    fast_spline = BezierSplineBuilder2D(initial_dir=np.array([1., -1.]) / norm_factor).add(BSB.LineSegment(sl)).instantiate_spline()
    fast_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(fast_spline, var, elevate_segments_degree=10)
    f_mean, f_cov = fast_pspline.gp_discretize(point_distance=0.8, sequence_length_cap=cap_len)

    acc_spline = BezierSplineBuilder2D(initial_dir=np.array([-1., -1.]) / norm_factor).add(BSB.LineSegment(sl)).instantiate_spline()
    acc_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(acc_spline, var, elevate_segments_degree=10)
    a_mean, a_cov = acc_pspline.gp_discretize(acceleration=0.1, init_point_distance=0.5, sequence_length_cap=cap_len)
    
    dec_spline = BezierSplineBuilder2D(initial_dir=np.array([-1., 1.]) / norm_factor).add(BSB.LineSegment(sl)).instantiate_spline()
    dec_pspline = ProbabilisticBezierSpline.approximate_uniform_variance(dec_spline, var, elevate_segments_degree=10)
    d_mean, d_cov = dec_pspline.gp_discretize(acceleration=-0.075, init_point_distance=1., sequence_length_cap=cap_len)
    
    return TrajectoryGMM(
        weights=[0.25, 0.25, 0.25, 0.25],
        means=[s_mean, f_mean, a_mean, d_mean],
        covs=[s_cov, f_cov, a_cov, d_cov],
        sampling_seed=123,
        name="diamond"
    )


def synth_hyang():
    """
    Walking-Path Structure imitating sdd-hyang:

    ######## 2 2 2 ########
    ########       ########
    ########       ########
    1                     3
    1                     3
    1                     3
    ########       ########
    ########       ########
    ########              4
    ########              4
    ########       ########
    ######## 5 5 5 ########

    Paths:
    1 -> 2
    1 -> 3
    1 -> 4
    1 -> 5
    2 -> 1
    2 -> 3
    2 -> 4
    2 -> 5
    3 -> 1
    3 -> 2
    3 -> 5
    4 -> 1
    4 -> 2
    5 -> 1
    5 -> 2
    5 -> 3
    ------
    = 16 paths, i.e. mixture components

    Other notes:
    - set spline endpoints a bit behind the actual sink in the scene, so that the posterior doesnt collapse as much into a single mean point (time horizon ends before the curve ends)
    - ...
    """
    paths = [
        # 1 -> 2
        BezierSplineBuilder2D(initial_dir=np.array([1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 1 -> 3
        BezierSplineBuilder2D(initial_dir=np.array([1., 0.])).add(BSB.LineSegment(3)).add(BSB.LineSegment(4)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 1 -> 4
        BezierSplineBuilder2D(initial_dir=np.array([1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 1 -> 5
        BezierSplineBuilder2D(initial_dir=np.array([1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(4)).instantiate_spline(),
        
        # 2 -> 1
        BezierSplineBuilder2D(origin=np.array([5., 5.]), initial_dir=np.array([0., -1.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 2 -> 3
        BezierSplineBuilder2D(origin=np.array([5., 5.]), initial_dir=np.array([0., -1.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 2 -> 4
        BezierSplineBuilder2D(origin=np.array([5., 5.]), initial_dir=np.array([0., -1.])).add(BSB.LineSegment(3)).add(BSB.LineSegment(4)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 2 -> 5
        BezierSplineBuilder2D(origin=np.array([5., 5.]), initial_dir=np.array([0., -1.])).add(BSB.LineSegment(3)).add(BSB.LineSegment(4)).add(BSB.LineSegment(4)).instantiate_spline(),
        
        # 3 -> 1
        BezierSplineBuilder2D(origin=np.array([10., 0.]), initial_dir=np.array([-1., 0.])).add(BSB.LineSegment(3)).add(BSB.LineSegment(4)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 3  > 2
        BezierSplineBuilder2D(origin=np.array([10., 0.]), initial_dir=np.array([-1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 3 -> 5
        BezierSplineBuilder2D(origin=np.array([10., 0.]), initial_dir=np.array([-1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(4)).instantiate_spline(),
        
        # 4 -> 1
        BezierSplineBuilder2D(origin=np.array([10., -4]), initial_dir=np.array([-1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 4 -> 2
        BezierSplineBuilder2D(origin=np.array([10., -4]), initial_dir=np.array([-1., 0.])).add(BSB.LineSegment(3)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(4)).add(BSB.LineSegment(3)).instantiate_spline(),
        
        # 5 -> 1
        BezierSplineBuilder2D(origin=np.array([5., -6.]), initial_dir=np.array([0., 1.])).add(BSB.LineSegment(4)).add(BSB.CurveSegment(90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 5 -> 2
        BezierSplineBuilder2D(origin=np.array([5., -6.]), initial_dir=np.array([0., 1.])).add(BSB.LineSegment(4)).add(BSB.LineSegment(4)).add(BSB.LineSegment(3)).instantiate_spline(),
        # 5 -> 3
        BezierSplineBuilder2D(origin=np.array([5., -6.]), initial_dir=np.array([0., 1.])).add(BSB.LineSegment(4)).add(BSB.CurveSegment(-90, 2, 2)).add(BSB.LineSegment(3)).instantiate_spline()
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
        #print(np.reshape(np.all(c - c.T == 0), [-1]), np.linalg.eigvals(c))   # TODO: weg
        #print(np.linalg.cholesky(c))  # TODO: weg
        print(f"comp {len(means)} length: {len(m)}")

    return TrajectoryGMM(
        weights=weights,
        means=means,
        covs=covs,
        sampling_seed=123,
        name="synth_hyang"
    )
