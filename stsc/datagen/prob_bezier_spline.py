from __future__ import annotations
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from stsc.datagen.bezier_spline import BezierSpline


class ProbabilisticBezierSpline:
    """
    Represents a probabilistic variant of a <BézierSpline>, which is defined by Gaussian control points.
    This builds on the framework of probabilistic Bézier curves [1,2,3].
  
    [1] Hug, Ronny, Wolfgang Hübner, and Michael Arens. "Introducing probabilistic bézier curves for n-step sequence prediction." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 06. 2020. 
    [2] Hug, Ronny. Probabilistic Parametric Curves for Sequence Modeling. Vol. 55. KIT Scientific Publishing, 2022.
    [3] Hug, Ronny, et al. "Bézier Curve Gaussian Processes." arXiv preprint arXiv:2205.01754 (2022).
    """
    def __init__(self, mean_spline: BezierSpline, segment_variances: List[np.ndarray]) -> None:
        """
        Initializes a Probabilistic Bézier Spline from a <BézierSpline> used as a mean function and a list of variances (or covariance matrices) for all segment control points.
        
        :param mean_spline: <BezierSpline> representing the mean curve.
        :param segment_variances: List of variance/covariance matrices for each segment. Each numpy array in the list corresponds to one spline segment and has shape [ n_control_points, dim[, dim] ]. If a vector is given, it is converted into a diagonal covariance matrix. 
        """
        assert mean_spline.num_segments == len(segment_variances), "Mismatch in number of segments."
        for i in range(mean_spline.num_segments):
            assert len(mean_spline.segments[i]) == len(segment_variances[i]), f"Mismatch in number of control point variances in segment '{i}'."

        self.mean_spline = mean_spline
        self.segment_means = mean_spline.segments
        self.segment_covariance_matrices = [np.array([np.diag(v) if len(v.shape) == 1 else v for v in seg_vars]) for seg_vars in segment_variances]

    @classmethod
    def approximate_uniform_variance(
        cls, 
        spline: BezierSpline, 
        target_point_variance: float,
        approx_curve_pts: int = 100,
        elevate_segments_degree: int = 5  # 0 -> no elevation
    ) -> ProbabilisticBezierSpline:
        """ 
        Instantiates a ProbabilisticBezierSpline such that points along the spline have approximately constant variance.
        Thereby, the variances are approximated on a per segment basis. Further, this function only considers isotropic covariance matrices for now.

        In order to provide more control points for curve point variance approximation, the degree of each spline segment might be elevated (without changing its shape).

        Note that as an alternative to this approximation (which can and will NOT result in perfectly equal variances), the approach presented in [1] could also be employed, where an adjusted Bernstein base for the Bézer curve is proposed.

        [1] Jørgensen, Martin, and Michael A. Osborne. "Bezier Gaussian Processes for Tall and Wide Data." Advances in Neural Information Processing Systems 35 (2022): 24354-24366.
    
        :param spline: The Bézier Spline to extend into a probabilistic Bézier spline
        :param target_point_variance: The target variance for each point
        :param approx_curve_pts: Number of curve points to base approximation off
        :param elevate_segments_degree: (integer) elevation degree; 0 is no elevation
        :return: Probabilistic Bézier Spline with isotropic covariance matrices
        """
        t_vals = np.linspace(0., 1., num=approx_curve_pts)
        new_segments = []
        segment_variances = []
        for segment in spline.segments:
            # (Optionally) elevate segment degree
            if len(segment) == 2:
                elev_segment = cls._elevate_line_segment(segment, elevate_segments_degree)
            else:
                elev_segment = cls._elevate_segment(segment, elevate_segments_degree)

            # use linearized curve construction for the optimization routine (use "Bernstein transformation matrix")
            # -> we are only working with isotropic covariance matrices
            # -> the curve point covariance matrix calculation is simplified to calculating the diagonal vector only
            # => calculate V_curve = B * V_ctrl,
            # => where B is the Bernstein transformation matrix with squared Bernstein coefficients and V is a stacked vector of variances containing all control/curve point variances.
            d = elev_segment.shape[-1]
            n = len(elev_segment) - 1
            B = np.zeros([approx_curve_pts * d, (n + 1) * d])
            for i, t in enumerate(t_vals):
                b_block = [np.eye(d) * b_val**2 for b_val in BezierSpline.get_bernstein_coeffs(t, n)]
                for j in range(n + 1):
                    B[2*i:2*i+d, 2*j:2*j+d] = b_block[j]
            
            # optimize for control point variances with the objective of getting approximately equal variances along the curve
            # note that equality is not possible due to how the curve point variance calculation works
            init_vars = target_point_variance * np.ones([n + 1])  # we actually need (n+1)*d, but being isotropic each control point variance is duplicated
            target_vars = target_point_variance * np.ones([approx_curve_pts * d])
            objective = lambda x: np.square((B @ np.square(x).repeat(2).reshape([-1, 1])).squeeze() - target_vars).mean()
            res = opt.minimize(objective, init_vars, method='nelder-mead', options={'xatol': 1e-8, 'maxiter': (n+1)*200})
            seg_vars = np.square(res.x).repeat(2).reshape([-1, 2])

            new_segments.append(elev_segment)
            segment_variances.append(seg_vars)
        return cls(mean_spline=BezierSpline(new_segments), segment_variances=segment_variances)

    @staticmethod
    def _elevate_line_segment(segment, elevate_degrees) -> np.ndarray:
        # In the line case, we can simply add equidistant control points along the line.
        if elevate_degrees == 0:
            return segment

        tmp_spline = BezierSpline([segment])
        t_vals = np.linspace(0., 1., num=elevate_degrees + 2)
        return np.array([tmp_spline.curve_point(t) for t in t_vals]) 

    @staticmethod
    def _elevate_segment(segment, elevate_degrees) -> np.ndarray:
        # A degree n Bézier curve is defined by n+1 control points.
        # Elevated Bézier curve control points are calculated as
        # Q_i = i / (n + 1) * P_i-1 + (1 - i / (n + 1)) * P_i,
        # where n is the current curve degree and i in [1, n]
        # The degree is elevated incrementally.
        if elevate_degrees == 0:
            return segment

        elev_seg = segment[:]
        elevate_iterations = elevate_degrees
        while elevate_iterations > 0:
            tmp = [elev_seg[0]]
            n = len(elev_seg) - 1
            for i in range(1, len(elev_seg)):
                new_pt = i / (n + 1) * elev_seg[i - 1] + (1 - i / (n + 1)) * elev_seg[i]
                tmp.append(new_pt)
            tmp.append(elev_seg[-1])
            elev_seg = tmp[:]
            elevate_iterations -= 1     

        return np.array(elev_seg)

    @staticmethod
    def gaussian_bezier_curve_point(t, mu_cpts, cov_cpts):
        """
        Calculates the Gaussian curve point (mean and covariance) at position t on a Gaussian Bézier spline segment (Gaussian Bézier curve) defined by the control points cpts.

        :param t: Positional parameter of the Gaussian Bézier curve
        :param mu_cpts: Mean vectors of the Gaussian Bézier curve control points
        :param cov_cpts: Covariance matrices of the Gaussian Bézier curve control points
        :param flat: return mu and cov as one flattened and concatenated vector
        :return: Gaussian 2D point on the curve.
        """
        n = len(mu_cpts) - 1
        coeffs = BezierSpline.get_bernstein_coeffs(t, n)
        mu = np.sum([coeffs[i] * mu_cpts[i] for i in range(len(mu_cpts))], axis=0)
        cov = np.sum([coeffs[i] ** 2 * cov_cpts[i] for i in range(len(cov_cpts))], axis=0)

        return mu, cov

    def plot(self, axis: Optional[plt.Axes] = None, show: bool = False, n_pts: int = 100):
        """
        Draws this spline into a pyplot axis.

        :param axis: The axis to draw into. If None, is set to pyplot.gca()
        :param show: Call pyplot.show()?
        :param n_pts: Number of Gaussian curve points to draw
        """
        pts_per_seg = int(n_pts // len(self.segment_means))
        t_vals = np.linspace(0, 1, num=pts_per_seg)
        mean_curve = [self.gaussian_bezier_curve_point(t, self.segment_means[0], self.segment_covariance_matrices[0])[0] for t in t_vals]
        for i in range(1, len(self.segment_means)):
            mean_curve += [self.gaussian_bezier_curve_point(t, self.segment_means[i], self.segment_covariance_matrices[i])[0] for t in t_vals[1:]]
        mean_curve = np.asarray(mean_curve)

        curve_covars = [self.gaussian_bezier_curve_point(t, self.segment_means[0], self.segment_covariance_matrices[0])[1] for t in t_vals]
        for i in range(1, len(self.segment_means)):
            curve_covars += [self.gaussian_bezier_curve_point(t, self.segment_means[i], self.segment_covariance_matrices[i])[1] for t in t_vals[1:]]
        curve_covars = np.asarray(curve_covars)
        
        if axis is None:
            axis = plt.gca()

        lines = axis.plot(mean_curve[:, 0], mean_curve[:, 1], "-")
        axis.plot(mean_curve[:1, 0], mean_curve[:1, 1], "ko")

        color = lines[0].get_color()
        for t in range(len(mean_curve)):
            self._confidence_ellipse(mean_curve[t], curve_covars[t], axis, 3, color, "none", 0.33)
            self._confidence_ellipse(mean_curve[t], curve_covars[t], axis, 2, color, "none", 0.66)
            self._confidence_ellipse(mean_curve[t], curve_covars[t], axis, 1, color, "none", 1.)

        if show:
            plt.show()

    @staticmethod
    def _confidence_ellipse(mean, cov, axis, n_std, edgecolor, facecolor, alpha):
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

    def gp_discretize(
        self, 
        num_discretization_pts: Optional[int] = None,
        point_distance: Optional[float] = None,
        acceleration: Optional[float] = None,
        init_point_distance: Optional[float] = 1.,
        sequence_length_cap: Optional[int] = None
    ): 
        """ 
        Calculates full stacked mean vector and gram matrix of induced Gaussian process prior for a given curve discretization approach. 
        This GP prior covers a sequence of n points, whilst also modeling the correlations between points.
        Modifies formulas from [1] by calculating curve points along a Bézier spline (composite Bézier curve) instead of a single curve.
        This results in minor technical modifications switching between segment control point sets when calculating the expectations.
        The discretization approach is determined by setting the optional function parameters accordingly.

        [1] Hug, Ronny, et al. "Bézier Curve Gaussian Processes." arXiv preprint arXiv:2205.01754 (2022).

        :param num_discretization_pts: The number of Gaussian curve points to extract. Curve points are spread evenly across the underlying spline.
        :param point_distance: The euclidean distance between subsequent Gaussian curve points. Curve points are extracted, such that the distance requirement holds, yielding n curve points.
        :param acceleration: The increase in distance between subsequent Gaussian curve points. Curve points are extracted, such that the gradually increasing distance requirement holds, yielding n curve points. This approach also required setting <init_point_distance>.
        :param init_point_distance: Initial curve point distance, which is then increased according to <acceleration>.
        :param sequence_length_cap: Caps the maximum number of curve points for distance and acceleration approaches. This is regardless of the remaining distance to the spline's last control point. 
        """
        assert num_discretization_pts is not None or point_distance is not None or acceleration is not None, "Either <num_discretization_pts>, <target_distance> or <acceleration> must be set."

        # Determine curve mean vectors and corresponding curve positional parameters (t)
        if num_discretization_pts is not None:
            mean_pts, t_vals = self.mean_spline._discretize_equi_pts(num_discretization_pts, ret_t_vals=True)
        elif point_distance is not None:
            mean_pts, t_vals = self.mean_spline._discretize_const_dist(point_distance, ret_t_vals=True)
            num_discretization_pts = len(t_vals)
        else:
            mean_pts, t_vals = self.mean_spline._discretize_accelerating(acceleration, init_point_distance, ret_t_vals=True)
            num_discretization_pts = len(t_vals)

        if sequence_length_cap is not None and sequence_length_cap > 0:
            mean_pts = mean_pts[:sequence_length_cap]
            t_vals = t_vals[:sequence_length_cap]

        # Build gram matrix using the N-Spline Gaussian process' covariance function
        b_map = lambda b,x: np.array([b[i] * x[i] for i in range(len(b))])
        gram_matrix = np.zeros(shape=[2*len(t_vals), 2*len(t_vals)])
        # build upper triangular and mirror it later for the lower triangle
        # -> K(t_i, t_j) = E[XY^T] - E[X*m^T_y] - E[m_x*Y^T] + m_x*m^T_y
        #                = E[XY^T] - m_x*m^T_y - m_x*m^T_y + m_x*m^T_y
        #                = E[XY^T] - m_x*m^T_y
        for i in range(len(t_vals)):
            for j in range(i, len(t_vals)):

                t_i = t_vals[i]
                t_j = t_vals[j]

                # get segment indices
                tau = 1 / (self.mean_spline.num_segments)
                a_i = min(int(t_i // tau), self.mean_spline.num_segments - 1) 
                a_j = min(int(t_j // tau), self.mean_spline.num_segments - 1)
                tc_i = (t_i - a_i * tau) / ((a_i + 1) * tau - a_i * tau)  # map current t onto corresponding segment t
                tc_j = (t_j - a_j * tau) / ((a_j + 1) * tau - a_j * tau)  
                P_i = self.mean_spline.segments[a_i]
                P_j = self.mean_spline.segments[a_j]
                N_i = len(P_i)
                N_j = len(P_j)
                b_i = BezierSpline.get_bernstein_coeffs(tc_i, N_i - 1)
                b_j = BezierSpline.get_bernstein_coeffs(tc_j, N_j - 1)     
                C_i = self.segment_covariance_matrices[a_i]

                # m_x m_y^T
                m_x = np.sum(b_map(b_i, P_i), axis=0).reshape([-1, 1])
                m_y = np.sum(b_map(b_j, P_j), axis=0).reshape([-1, 1])

                # E[X Y^T]
                # cases (a_j >= a_i always holds):
                # - a_i == a_j -> inter-curve correlation exists, calculate like in normal N-Curve
                # - a_j == a_i + 1 -> correlation between last control points in i and first control points in j if c1 or c2 continuity is given
                # - a_j > a_i + 1 -> there is no correlation and expectation separates in 2 independent terms (E[X Y^T] = E[X] E[Y^T])
                if a_i == a_j:  # P_i == P_j
                    tmp = []
                    for k in range(N_i):
                        for l in range(N_i):
                            if k == l:  # b_i[k] * b_j[k] * (Cov_k + mu_k @ mu^T_k)
                                tmp.append(b_i[k] * b_j[k] * (C_i[k] + P_i[k].reshape([2, 1]) @ P_j[k].reshape([1, 2])))
                            else:  # b_i[k] * b_j[l] * mu_k @ mu^T_l
                                tmp.append(b_i[k] * b_j[l] * P_i[k].reshape([2, 1]) @ P_j[l].reshape([1, 2]))
                    E_xy = np.sum(tmp, axis=0)
                elif a_j == a_i + 1: 
                    # check for continuity > 0 (c0 continuity is a given)
                    # - check if i's last offset vector is equal to j's first offset vector
                    # - use a determinant check for that for numerical stability
                    # -> x1y2 = x2y1 => parallel (if both vectors form a matrix of column vectors, the determinant is 0 in the case of parallel vectors)
                    i_off = P_i[-1] - P_i[-2]
                    j_off = P_j[1] - P_j[0]
                    if np.isclose(i_off[0] * j_off[1], i_off[1] * j_off[0]):  # parallel: c1- or c2-continuity
                        # calc length scale factor
                        s = np.linalg.norm(j_off) / np.linalg.norm(i_off)
                        tmp = []
                        for k in range(N_i):
                            for l in range(N_j):
                                if l == 0 and k == N_i - 1:  # connecting point 
                                    # same as before (b_i[k] * b_j[k] * (Cov_k + mu_k @ mu^T_k))
                                    tmp.append(b_i[-1] * b_j[0] * (C_i[-1] + P_i[-1].reshape([2, 1]) @ P_j[0].reshape([1, 2])))
                                
                                # for the following two: P_j[1] = P_i[-1] + s * (P_i[-1] - P_i[-2])
                                elif l == 1 and k == N_i - 2:  
                                    # -> E[b1*b2*P_i[-2]*P_j[1]] 
                                    # = b1*b2*E[P_i[-2] * (P_i[-1] + s * (P_i[-1] - P_i[-2]))]
                                    # = b1*b2*E[P_i[-2] * P_i[-1] + s * P_i[-2] * (P_i[-1] - P_i[-2])]
                                    # = b1*b2 * ( E[P_i[-2] * P_i[-1]] + s * E[P_i[-2] * P_i[-1] - P_i[-2] * P_i[-2]] )
                                    # = b1*b2 * ( E[P_i[-2] * P_i[-1]] + s * E[P_i[-2] * P_i[-1]] - s * E[P_i[-2] * P_i[-2]] )
                                    # = b1*b2 * ( E[P_i[-2]] * E[P_i[-1]] + s * E[P_i[-2]] * E[P_i[-1]] - s * E[P_i[-2] * P_i[-2]])
                                    b1b2 = b_i[-2] * b_j[1]
                                    t1 = P_i[-2].reshape([2, 1]) @ P_i[-1].reshape([1, 2])  # E[P_i[-2]] * E[P_i[-1]]
                                    t2 = P_i[-2].reshape([2, 1]) @ P_i[-2].reshape([1, 2])  # E[P_i[-2] * P_i[-2]]
                                    tmp.append(b1b2 * (t1 + s*t1 - s*t2))
                                elif l == 1 and k == N_i - 1: 
                                    # -> E[b1*b2*P_i[-1]*P_j[1]] 
                                    # = b1*b2 * ( E[P_i[-1] * P_i[-1]] + s * E[P_i[-1] * P_i[-1]] - E[P_i[-1] * P_i[-2]] )
                                    # = b1*b2 * ( E[P_i[-1] * P_i[-1]] + s * E[P_i[-1] * P_i[-1]] - E[P_i[-1]] * E[P_i[-2]] )
                                    b1b2 = b_i[-1] * b_j[1]
                                    t1 = P_i[-1].reshape([2, 1]) @ P_i[-1].reshape([1, 2])  # E[P_i[-1] * P_i[-1]]
                                    t2 = P_i[-1].reshape([2, 1]) @ P_i[-2].reshape([1, 2])
                                    tmp.append(b1b2 * (t1 + s*t1 - s*t2))
                                else:  # independent control points
                                    # e.g. E[b1*b2*P_i[0]*P_j[0]]
                                    # = b1*b2 * E[P_i[0]*P_j[0]]
                                    # = b1*b2 * E[P_i[0]] * E[P_j[0]]
                                    tmp.append(b_i[k] * b_j[l] * P_i[k].reshape([2, 1]) @ P_j[l].reshape([1, 2]))
                        E_xy = np.sum(tmp, axis=0)  
                    else:  # not parallel: c1-continuity only
                        print("not parallel!")
                        tmp = []
                        for k in range(N_i):
                            for l in range(N_j):
                                if l == 0 and k == N_i - 1:  # connecting point 
                                    tmp.append(b_i[k] * b_j[k] * (C_i[k] + P_i[k].reshape([2, 1]) @ P_j[k].reshape([1, 2])))
                                else:
                                    tmp.append(b_i[k] * b_j[l] * P_i[k].reshape([2, 1]) @ P_j[l].reshape([1, 2]))
                        E_xy = np.sum(tmp, axis=0)
                else:  # a_j > a_i + 1  
                    # expectation decomposes into independent expectations
                    tmp = []
                    for k in range(N_i):
                        for l in range(N_j):
                            tmp.append(b_i[k] * b_j[l] * P_i[k].reshape([2, 1]) @ P_j[l].reshape([1, 2]))
                    E_xy = np.sum(tmp, axis=0)

                K_xy = E_xy - m_x @ m_y.T 
                gram_matrix[2*i:2*i+2, 2*j:2*j+2] = K_xy
        gram_matrix = np.triu(gram_matrix) + np.triu(gram_matrix, k=1).T
        return mean_pts.reshape([-1]), gram_matrix
