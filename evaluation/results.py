from typing import List, Dict, Union

import numpy as np
from evaluation.test_case import TestCase
from utils.misc import copy_data


class ResData(object):
    def __init__(self, model_name: str, model_output: Union[np.ndarray, List[np.ndarray]], test_case: TestCase, test_results: Dict[str, List[float]]):
        self.model_name = model_name
        self.model_output = copy_data(model_output)
        self.test_case_uuid = test_case.uuid
        self.test_results = test_results
        self.is_baseline = False

    """@classmethod
    def from_file(cls, file_path):
        raw_data = io.decompress_data(file_path)
        sanity_reults = [np.array(tr) for tr in raw_data["sanity_test_results"]]
        test_results = {k: [np.array(tr) for tr in v] for k, v in raw_data["test_results"].items()}
        return _ResData(Tasks(raw_data["task"]), raw_data["test_case_num"], raw_data["model_name"], sanity_reults, test_results)"""