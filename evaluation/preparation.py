# class ResultWriter -> takes testcase(s) and (a) model(s) with call signature (need to define stsc_call?) trajs_in -> trajs_out (or distr_out, depending test_case) -> iterates evaluation set and dumps results to a standardized file format
from utils.dataset import Datasets
from evaluation.properties import Tasks, LengthMode, ProcessingMode, Target
from evaluation.test_case import TestCase
import utils.io as io
from utils.sequence_processing import SequenceDecomposer


class ResultWriter(object):
    def __init__(self, out_file_path: str):
        self.file_path = out_file_path
        if not self.file_path.endswith(".gz"):
            self.file_path += ".gz"

    def __call__(self, test_case: TestCase, model):
        assert hasattr(model, "stsc_name") and callable(getattr(model, "stsc_name")), "Model is missing function <stsc_name>."  # stsc_name should return the models name used for benchmark output
        assert hasattr(model, "stsc_call") and callable(getattr(model, "stsc_call")), "Model is missing function <stsc_call>."  # stsc_call should take 1 trajectory as a sequence of inputs (to simulate different processing modes)

        model_out = []
        for in_data, _ in test_case.eval_data_iterator():
            decomposed_in = SequenceDecomposer.decompose(in_data, test_case.window_size)
            model_out.append(model.stsc_call(decomposed_in).tolist())
        io.compress_data({"model_name": model.stsc_name(), "model_output": model_out, "test_case": test_case.uuid}, self.file_path)

# TODO: der slidingwindow iterator gibt nur vor wie die eingabe verarbeitet werden muss, das target ist trotzdem immer die ganze (teil-)sequenz -> iterator.decompose_sequence
