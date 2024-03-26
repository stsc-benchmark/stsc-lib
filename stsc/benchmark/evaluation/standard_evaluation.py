from stsc.benchmark.evaluation.factors.evaluation_factor import EvaluationFactor
from stsc.benchmark.evaluation.factors.variance_estimation import VarianceEstimation
from stsc.benchmark.evaluation.factors.outlier_robustness import OutlierRobustness
from stsc.benchmark.evaluation.factors.responsiveness import Responsiveness
from stsc.benchmark.evaluation.factors.overall_performance import OverallPerformance


class StandardEvaluation:
    @classmethod
    def all_tests(cls) -> EvaluationFactor:
        return [
            VarianceEstimation(),
            OutlierRobustness(),
            Responsiveness(),
            OverallPerformance()
        ]


