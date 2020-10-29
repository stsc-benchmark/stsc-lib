import os
import json
import gzip
from datetime import datetime
from enum import Enum
from typing import List, Dict, Union, Iterator, Optional

import numpy as np
import matplotlib.pyplot as plt

#from utils.training_helper import TrainingHelper
#from evaluation.tasks import Tasks
#from evaluation.test_cases import TestCases, TestCase
from evaluation.test_case import TestCase
from evaluation.baselines import Baselines
from evaluation.metrics import Metric, Metrics
#from evaluation.evaluate import evaluate
from evaluation.visualize import VisOption
from evaluation.results import ResData
import utils.io as io


class Benchmark(object):  # modeled as a state machine (auch eine funktion sodass alle test cases durchgerattert werden?)
    class VisMode(Enum):
        SHOW = 0
        SAVE = 1

    def __init__(self):
        self.benchmark_root = io.module_path()
        self._reset_lists()

    def _reset_lists(self):
        self._baseline_results = []
        self._comp_model_results = []
        self._metrics: List[Metric] = []
        self._vis_options: List[VisOption] = []
        self._vis_mode: Benchmark.VisMode = Benchmark.VisMode.SHOW
        self._vis_mode_save_dir: Optional[str] = None

    """def add_baselines(self):
        for baseline in self.current_test_case().available_baselines:
            self._baseline_results.append(os.path.join(baseline.value, self.current_test_case().tag + ".gz"))"""

    def add_baseline_comparison(self, baseline: Baselines):  # TODO: only add placeholder that loads specific results for each testcase at hand
        if baseline not in self.__tc.available_baselines:
            print(f"Warning: Did not add baseline (incompatible baseline, use one of: {[b.name for b in self.__tc.available_baselines]})")
            return

        self._baseline_results.append({"name": baseline.name, "path": os.path.join(baseline.value, self.current_test_case().tag + ".gz")})
        #self._baseline_results.append(os.path.join(baseline.value, self.current_test_case().tag + ".gz"))

    def add_comparison(self, file_path: str):
        self._comp_model_results.append(file_path)

    def add_comparisons(self, file_path_list: List[str]):
        self._comp_model_results.extend(file_path_list)

    def add_comparison_from_directory(self, directory: str, model_name: str):
        file_path = os.path.join(directory, f"{self.__tc.tag}__{model_name}.gz")
        if os.path.exists(file_path):
            self._comp_model_results.append(file_path)
        else:
            print(f"Warning: Couldn't find comparison model: {file_path}")

    def add_metrics(self):  # TODO: adds all metrics
        print("hi")

    def add_metric(self, metric: Union[Metric, Metrics]):  # TODO: adds specific metric
        if isinstance(metric, Metrics):
            self._metrics.append(metric.value)
        else:
            self._metrics.append(metric)

    def add_output(self):  # output: console oder file
        print("hi")

    def add_output_formatter(self):
        print("hi")

    def add_visualization_option(self, vis_opt: VisOption):
        self._vis_options.append(vis_opt)

    def set_visualization_mode(self, mode: VisMode, save_dir: Optional[str] = None, dir_timestamp: bool = False):
        if mode is Benchmark.VisMode.SAVE and save_dir is None:
            print("Warning: Cannot set visualization mode to <SAVE> without specifiying <save_dir>!")
            return

        self._vis_mode = mode
        time_stamp = "__" + datetime.now().strftime("%Y-%m-%d_%H-%M") if dir_timestamp else ""
        self._vis_mode_save_dir = None if save_dir is None else os.path.join(save_dir + time_stamp, self.current_test_case().tag)

    def run(self):
        res_data = self._run_eval()
        self._run_visualization(res_data)

    def _run_eval(self) -> List[ResData]:  # TODO: das _ResData Konstrukt kann man schon beibehalten => muss etwas umstrukturiert werden, dass das auch klarer ist für die visualisierung => bei den metriken nur pro-trajektorie metriken anbieten und in der übersicht dann einfach mitteln -> "mean-metric"
        res_data: List[ResData] = []

        for model_data_path in self._comp_model_results:
            model_data = io.decompress_data(model_data_path)
            output = np.array(model_data["model_output"])
            test_case = TestCase.from_uuid(model_data["test_case"])

            res_data.append(ResData(
                model_name=model_data["model_name"],
                model_output=output,
                test_case=test_case,
                test_results={metric.abbreviation: metric(output, test_case.eval_output()) for metric in self._metrics}
            ))

            for metric, res in res_data[-1].test_results.items():
                print(f"Mean {metric}: {np.mean(res)}")

        return res_data


        """res_data: List[_ResData] = []

        print("Running sanity test")
        for baseline in self._baseline_results:
            print("  Baseline:", baseline["name"])
            rd = _ResData.from_file(baseline["path"])
            if not self._test_res_data(rd):
                print(f"    Warning: Cannot evaluate {rd.model_name} in current test case (incompatible).")
                continue
            rd.model_name = baseline["name"]
            rd.is_baseline = True

            for metric in self._metrics:
                print(f"    {metric.name}: {metric(rd.sanity_test_results, self.__tc.sanity_test_set().output)}")

            res_data.append(rd)

        print()
        print("Running tests")
        for i, test in enumerate(self.__tc.test_sets()):
            print(f"{i + 1} {test.name}...")
            for eval_model in res_data:
                print(f"  {'Baseline' if eval_model.is_baseline else 'Model'}:", eval_model.model_name)

                for metric in self._metrics:
                    print(f"    {metric.name}: {metric(eval_model.test_results[test.name], test.output)}")

        return res_data"""

    """
    res_data.append(_ResData(
                model_name=model_data["model_name"],
                model_output=output,
                test_case=test_case,
                test_results={metric.name: metric(output, test_case.eval_output()) for metric in self._metrics}
            ))
    """
    def _run_visualization(self, res_data: List[ResData]):
        if self._vis_mode is not Benchmark.VisMode.SAVE:
            self._vis_mode_save_dir = None

        for vis_opt in self._vis_options:
            vis_opt(res_data, save_dir=self._vis_mode_save_dir)
