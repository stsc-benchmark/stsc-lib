from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import scipy.optimize as opt


class DiscretizationApproach(Enum):
    EQUIDISTANT_POINTS = 0  # n points are spread evenly along the curve
    CONSTANT_DISTANCE = 1  # distance between curve points is pre-defined (and number of curve points are dependent on distance)
    CONSTANT_ACCELERATION = 2  # distance between curve points increase/decreases with a pre-defined rate
    #CONSTANT_CURVE_STEP = 3  # points along the curve are chosen according to evenly spread curve positions t in [0, 1]
    # -> CONSTANT_CURVE_STEP is not implemented yet


class BezierSpline:
    """
    Represents a composite Bézier curve (i.e. a Bézier-Spline) consisting of n >= 1 segments with d-dimensional control points.
    Provides several functions for altering, discretizing and plotting Bézier-Splines.
    """
    def __init__(self, segments: Optional[List[np.ndarray]] = None, pt_dim: Optional[int] = None) -> None:
        """
        Initializes a Bézier-Spline using either an initial list of segments or the spline's dimensionality (point dimension).

        :param segments: Optional list of segments. A segment is defined as a numpy array with shape [ n_control points, point dim ]
        :param pt_dim: Optional point dimension.
        """
        assert segments is not None or pt_dim is not None, "Either parameter needs be not None."
        if segments is not None:
            assert len(segments) > 0, "List of segments must not be empty."

        self._pt_dim = segments[0].shape[-1] if segments is not None else pt_dim

        if segments is not None:
            self._segments = segments[:]
        else:
            self._segments = None

    @staticmethod
    def get_bernstein_coeffs(t: float, n: int) -> np.ndarray:
        """
        Calculates the values of the bernstein polynomials given the Bézier curve degree and a specific value for t.

        :param t: Positional parameter of the Bézier curve
        :param n: Degree of the Bézier curve
        :return: Array of mixing coefficients for the Bézier curve control points.
        """
        return np.array([comb(n, i) * (1 - t) ** (n - i) * t ** i for i in range(n + 1)])


    @staticmethod
    def bezier_curve_point(t: float, cpts: np.ndarray) -> np.ndarray:
        """
        Calculates the point at position t on a Bézier curve defined by respective control points.

        :param t: Positional parameter of the Bézier curve
        :param cpts: Bézier curve control points
        :return: Point on the curve.
        """
        n = len(cpts) - 1
        coeffs = BezierSpline.get_bernstein_coeffs(t, n)
        return np.sum([coeffs[i] * cpts[i] for i in range(len(cpts))], axis=0)

    @property
    def segments(self) -> List[np.ndarray]:
        return [np.copy(e) for e in self._segments]  # returns a copy, as the original list must not be modified

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    def add_segment(self, segment: Union[List,np.ndarray]) -> BezierSpline:
        """
        Adds a Bézier curve segment extending this Bézier-Spline.
        This is an in-place operation and returns self.

        :param segment: The new segment to add
        :return: self
        """
        if type(segment) is list:
            segment = np.asarray(segment)

        assert len(segment.shape) == 2, "Segment array must be in shape [n_points, point_dim]."
        assert segment.shape[-1] == self._pt_dim

        if self._segments is None:
            self._segments = [segment]
            return self
        
        self._segments.append(segment)
        return self

    def curve_point(self, pos_t: float) -> np.ndarray:
        """
        Calculates a curve point given the position t in [0, 1] on the spline curve.
        First maps t onto the corresponding Bézier curve segment and then calculates the curve point according to that segment
        using the <bezier_curve_point> function.

        :param pos_t: positional parameter
        :return: curve point
        """
        assert 0. <= pos_t <= 1., f"<pos_t> must be in [0, 1]. Given: {pos_t}"

        # map spline curve position onto local curve position
        n_curves = len(self._segments)
        tmp = np.linspace(0, 1, n_curves + 1)
        local_curve_bounds = list(zip(tmp[:-1], tmp[1:]))

        # finds first interval given pos_t is contained in
        local_index = np.argmax([b[0] <= pos_t <= b[1] for b in local_curve_bounds])
        local_bounds = local_curve_bounds[local_index]
        local_pos = (pos_t - local_bounds[0]) / (local_bounds[1] - local_bounds[0])

        return self.bezier_curve_point(local_pos, self._segments[local_index])

    def plot(
        self,
        num_pts: int = 100, 
        connect_control_pts: bool = False,
        highlight_start: bool = False, 
        show: bool = True
    ):
        """
        Plots this spline into the currently active pyplot graph (plt.gca()).

        :param num_pts: Number of curve points to draw
        :param connect_control_pts: connect control points with lines to indicate control point sequence and convex hull
        :param highlight_start: highlights the first control points
        :param show: Flag to call plt.show() or not
        """
        ax = plt.gca()
        t_vals = np.linspace(0, 1, num=num_pts)
        for i, seg in enumerate(self._segments):
            seg_pts = np.array([self.bezier_curve_point(t, seg) for t in t_vals])
            ax.plot(seg_pts[:, 0], seg_pts[:, 1], "-")
            if connect_control_pts:
                ax.plot(seg[:, 0], seg[:, 1], "ko--", alpha=0.5)
            else:
                ax.plot(seg_pts[:1, 0], seg_pts[:1, 1], "ko")

            if highlight_start and i == 0:
                ax.plot(seg[:1, 0], seg[:1, 1], "ko", markerfacecolor="none", markersize=10)

        if show:
            plt.show()

    def discretize(
        self, 
        n_pts: Optional[int] = None, 
        target_distance: Optional[float] = None, 
        init_distance: Optional[float] = None,
        target_acceleration: Optional[float] = None,
        ret_t_vals: bool = False,
        approach: DiscretizationApproach = DiscretizationApproach.EQUIDISTANT_POINTS,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Discretize this Bézier-Spline by extracting curve points according to a given approach.
        
        :param n_pts: Number of points to extract for <equidistant points> approach
        :param target_distance: Target inter-point distance for <constant distance> approach
        :param ret_t_vals: Flag indicating whether to return the corresponding t values
        :param approach: The discretization approach
        :param verbose: Verbosity flag
        :return: Extracted curve points as numpy array 
        """
        if approach is DiscretizationApproach.EQUIDISTANT_POINTS:
            assert n_pts is not None, "<n_pts> is required for equidistant points discretization."
            return self._discretize_equi_pts(n_pts, ret_t_vals, verbose)

        if approach is DiscretizationApproach.CONSTANT_DISTANCE:
            assert target_distance is not None, "<target_distance> is required for constant distance discretization."
            return self._discretize_const_dist(target_distance, ret_t_vals, verbose)
        
        if approach is DiscretizationApproach.CONSTANT_ACCELERATION:
            assert init_distance is not None and target_acceleration is not None, ""
        
        # TODO: per-segment approaches allowing for accelerating/decelerating segments in-between <- for paper: not supported right now

    def _discretize_equi_pts(self, n_pts: int, ret_t_vals: bool = False, verbose: bool = False) -> np.ndarray:
        """ 
        Calculates curve points, such that the distance between subsequent curve points is constant. 
        Here, the first and last curve points are fixed to correspond to the first and last control points of the Spline, respectively.
        """
        if verbose:
            print("Calculating spline discretization <equidistant points>...")

        # use equidistant t_vals as initial values for optimization
        t_vals = np.linspace(0, 1, num=n_pts)
        # use optimization algo with bounds constraints only -> choose lower/upper bounds as the mean of subsequent initial t_vals
        half_step = 0.5 * (t_vals[1] - t_vals[0])
        bounds = opt.Bounds(lb=[t - half_step for t in t_vals[1:-1]], ub=[t + half_step for t in t_vals[1:-1]], keep_feasible=True)  # 0 and 1 are fixed

        res = opt.minimize(self.__discr_objective, t_vals[1:-1], method='trust-constr',  
                           jac="2-point", hess=opt.SR1(),options={'verbose': verbose}, 
                           bounds=bounds)

        t_vals_opt = [0.] + list(res.x) + [1.]
        pts = np.array([self.curve_point(t) for t in t_vals_opt])

        if verbose:
            print(f"last err: {self.__discr_objective(res.x)}")
            print("dists:", np.linalg.norm(pts[1:] - pts[:-1], axis=1))
            print()

        if ret_t_vals:
            return pts, t_vals_opt
        return pts

    def __discr_objective(self, x):
        x_lst = [0.] + list(x) + [1.]
        pts = np.array([self.curve_point(t) for t in x_lst])
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mdist = np.mean(dists)

        return np.square(dists - mdist).mean()

    def _discretize_const_dist(
        self, 
        dist: float, 
        ret_t_vals: bool = False, 
        verbose: bool = False
    ) -> np.ndarray:
        """ 
        Calculates curve points, such that the distance between subsequent curve points complies with a given distance. 
        Here, the first curve point is fixed to correspond to the first control points of the spline.
        """       
        tmp = np.array([self.curve_point(t) for t in np.linspace(0., 1., num=10)])
        approx_curve_len = np.sum(np.linalg.norm(tmp[1:] - tmp[:-1], axis=1))
        default_delta = 1. / (approx_curve_len / dist)
        selected_t_vals = [0.]
        pts = [self.curve_point(0.)]
        last_pt = self.curve_point(1.)

        # Uses an interval search style algorithm to sequentially find curve points approximately complying with the given distance requirement 
        while np.linalg.norm(last_pt - pts[-1]) >= dist:
            t_search = selected_t_vals[-1] + default_delta
            delta = default_delta
            forward_search = True

            while True:
                t_search = min(t_search, 1.)
                pt_search = self.curve_point(t_search)
                dist_search = np.linalg.norm(pt_search - pts[-1])
                if np.isclose(dist_search, dist):
                    selected_t_vals.append(t_search)
                    pts.append(pt_search)
                    break

                if dist_search > dist and forward_search:
                    delta /= 2
                    forward_search = False
                    t_search -= delta
                elif dist_search > dist and not forward_search:
                    t_search -= delta
                elif dist_search < dist and forward_search:
                    t_search += delta
                elif dist_search < dist and not forward_search:
                    delta /= 2
                    forward_search = True
                    t_search += delta
                    
        if ret_t_vals:
            return np.array(pts), selected_t_vals
        return np.array(pts)

    def _discretize_accelerating(
        self, 
        acceleration: float, 
        init_target_distance: float,
        ret_t_vals: bool = False, 
        verbose: bool = False
    ) -> np.ndarray:
        """ 
        Calculates curve points, such that the distance between subsequent curve points complies with a given distance increasing at a given rate. 
        Here, the first curve point is fixed to correspond to the first control points of the spline.
        """    
        t_vals = np.linspace(0, 1, num=1000)
        selected_t_vals = [0.]
        pts = [self.curve_point(0.)]
        target_dist = init_target_distance
        
        for t in t_vals[1:]:
            cur_pt = self.curve_point(t)
            if np.linalg.norm(cur_pt - pts[-1]) >= target_dist:
                selected_t_vals.append(t)
                pts.append(cur_pt)
                target_dist += acceleration
                if acceleration < 0 and target_dist < abs(acceleration):
                    target_dist = abs(acceleration)
        if ret_t_vals:
            return np.array(pts), selected_t_vals
        return np.array(pts)


class BezierSplineBuilder2D:
    """ 
    Helper class for building Bézier-Splines using configurable segments commonly encountered in trajectory datasets, i.e. line segments and curved segments.
    """
    def __init__(self, origin: np.ndarray = np.zeros(2), initial_dir: np.ndarray = np.array([0., 1.])) -> None:
        """
        Initialize builder using information about the origin of the spline.

        :param origin: The origin point of the first spline segment
        :param initial_dir: The direction the spline is "starting", i.e. the gradient at the first control point. 
        """
        self._init_pos = [origin]
        self._init_offset = np.copy(initial_dir)
        self._segments = []
    
    def add(self, segment: Segment) -> BezierSplineBuilder2D:
        """ 
        Adds a line or curve segment to the builder's segments list. 
        Returns self in order to enable chaining of this function.
        """
        offsets = segment.offset_vector(self._init_offset)
        seg_pts = np.cumsum(self._init_pos + offsets, axis=0)

        self._segments.append(seg_pts)
        self._init_pos = [seg_pts[-1]]
        self._init_offset = seg_pts[-1] - seg_pts[-2]

        return self

    def instantiate_spline(self) -> BezierSpline:
        return BezierSpline(self._segments)

    class Segment:
        """ Segment base class """
        def offset_vector(self, prev_seg_offset: np.ndarray) -> List:
            """ Aligns segment offset vectors with previous segment control points and returns """
            raise Exception("<offset_vector> not implemented for subclass!")

        @staticmethod
        def _directed_angle_diff_2d(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
            dot = np.inner(vec_a, vec_b)
            det = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
            return np.arctan2(det, dot)

        @staticmethod
        def _rot_matrix_2d(rad: float) -> np.ndarray:
            return np.reshape([np.cos(rad), -np.sin(rad), np.sin(rad), np.cos(rad)], [2, 2])

    class LineSegment(Segment):
        def __init__(self, extent: float) -> None:
            """ 
            'Forward' moving segment defined by a single offset vector.
            Default movement is along the positive y-axis.
            """
            self._extent = extent

        def offset_vector(self, prev_seg_offset: np.ndarray) -> List:
            norm_vec = prev_seg_offset / np.linalg.norm(prev_seg_offset)
            return [self._extent * norm_vec]
        
    # TODO: add accelerating/decelerating segment -> SpeedChangeLineSegment. Also provide acceleration on curve segments?

    class CurveSegment(Segment):
        def __init__(self, angle: float, longitudinal_extent: float, lateral_extent: float) -> None:
            """
            Curve segment defined by 2 control points as offsets from previous segment's last control point.
            
            angle: Defines the deviation angle from a straight line and thus the change in direction achieved by this segment (Note that rotation is counter-clockwise)
            longitudinal_extent: Defines the distance traveled forward through this segment
            lateral_extent: Defines the distance traveled to the left/right through this segment
            """
            rad = np.deg2rad(angle)
            r = self._rot_matrix_2d(rad)
            self._lon_vec = np.array([0, longitudinal_extent])
            self._lat_vec = r @ np.array([0, lateral_extent])

        def offset_vector(self, prev_seg_offset: np.ndarray) -> List:
            rad = self._directed_angle_diff_2d(self._lon_vec, prev_seg_offset)
            r = self._rot_matrix_2d(rad)
            return [
                r @ self._lon_vec,
                r @ self._lat_vec
            ]
