import os
from typing import List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from evaluation.test_case import TestCase
from evaluation.results import ResData


# TODO: histogram option wäre noch gut


class VisOption(object):
    @classmethod
    def _gen_dir(cls, dir_str):
        if dir_str is None:
            return

        if not os.path.exists(dir_str):
            os.makedirs(dir_str)

    def __call__(self, res_data: List[ResData], save_dir=None):
        raise NotImplementedError("<call> not implemented for subclass!")


class Datasets(VisOption):
    def __init__(self, max_samples=None, plot_train=False):
        self._max_samples = max_samples
        self._plot_train = plot_train

    def __call__(self, res_data: List[ResData], save_dir=None):
        self._gen_dir(save_dir)

        for res in res_data:
            test_case = TestCase.from_uuid(res.test_case_uuid)
            plt.figure()
            plt.title(f"{test_case.dataset_name}: {test_case.task_name}")
            if self._plot_train:
                self._plot_dataset(test_case.get_raw_training_data()[0], color="b", label="train (input)")
            self._plot_dataset(test_case.get_raw_test_data()[0], color="g", label="test (input)")

            lims = test_case.plot_limits
            plt.legend()
            plt.xlim(lims[0])
            plt.ylim(lims[1])

            if save_dir is None:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/Datasets__{test_case.dataset_name}-{test_case.task_name}.png", dpi=300, format="png", bbox_inches='tight')
                plt.close()

    def _plot_dataset(self,  data, color="g", label="test"):
        for i, tr in enumerate(data[:self._max_samples]):
            if i == 0:
                plt.gca().plot(tr[:, 0], tr[:, 1], color + "d-", label=label)
            else:
                plt.gca().plot(tr[:, 0], tr[:, 1], color + "d-")
            plt.gca().plot(tr[:1, 0], tr[:1, 1], "ko", markersize=10, markerfacecolor="none")


class MaxErrors(VisOption):
    def __call__(self, res_data: List[ResData], save_dir=None):
        self._gen_dir(save_dir)

        for res in res_data:
            test_case = TestCase.from_uuid(res.test_case_uuid)
            test_data = test_case.eval_data()
            for metric, errs in res.test_results.items():
                plt.figure()
                plt.title(f"{res.model_name} - {test_case.dataset_name} - {test_case.task_name}: max{{{metric}}}")
                err_list = sorted([(i, err) for i, err in enumerate(errs)], key=lambda x: x[1], reverse=True)

                in_tr, out_tr = test_data[err_list[0][0]]
                plt.gca().plot(in_tr[:, 0], in_tr[:, 1], "gx--", label="input")
                plt.gca().plot(in_tr[:1, 0], in_tr[:1, 1], "ko", markersize=10, markerfacecolor="none")
                plt.gca().plot(out_tr[:, 0], out_tr[:, 1], "bd-", label="pred")
                plt.gca().plot(out_tr[:1, 0], out_tr[:1, 1], "ko", markersize=10, markerfacecolor="none")
                plt.gca().plot(out_tr[:1, 0], out_tr[:1, 1], "k.", markersize=1, markerfacecolor="none", label=f"err: {err_list[0][1]:.4f}")

                lims = test_case.plot_limits
                plt.legend()
                plt.xlim(lims[0])
                plt.ylim(lims[1])

                if save_dir is None:
                    plt.show()
                else:
                    plt.savefig(f"{save_dir}/MaxErrors__{res.model_name}-{test_case.dataset_name}-{test_case.task_name}-{metric}.png", dpi=300, format="png", bbox_inches='tight')
                    plt.close()


class Examples(VisOption):
    def __init__(self, n_examples):
        self.n_examples = n_examples

    def __call__(self, res_data: List[ResData], save_dir=None):
        self._gen_dir(save_dir)

        for res in res_data:
            test_case = TestCase.from_uuid(res.test_case_uuid)
            test_data = test_case.eval_data()
            for metric, errs in res.test_results.items():
                indices = np.linspace(0, len(test_data) - 1, num=self.n_examples).astype(np.int32)
                for i, index in enumerate(indices):
                    plt.figure()
                    plt.title(f"{res.model_name} - {test_case.dataset_name} - {test_case.task_name}: Example {i+1}")

                    in_tr, out_tr = test_data[index]
                    plt.gca().plot(in_tr[:, 0], in_tr[:, 1], "gx--", label="input")
                    plt.gca().plot(in_tr[:1, 0], in_tr[:1, 1], "ko", markersize=10, markerfacecolor="none")
                    plt.gca().plot(out_tr[:, 0], out_tr[:, 1], "bd-", label="pred")
                    plt.gca().plot(out_tr[:1, 0], out_tr[:1, 1], "ko", markersize=10, markerfacecolor="none")

                    lims = test_case.plot_limits
                    plt.legend()
                    plt.xlim(lims[0])
                    plt.ylim(lims[1])

                    if save_dir is None:
                        plt.show()
                    else:
                        plt.savefig(f"{save_dir}/Examples__{res.model_name}-{test_case.dataset_name}-{test_case.task_name}-{metric}__{str(i+1).zfill(len(str(i)))}.png", dpi=300, format="png", bbox_inches='tight')
                        plt.close()


class ErrorsColorGrade(VisOption):
    def __init__(self, max_samples=None):
        self._max_samples = max_samples

    def __call__(self, res_data: List[ResData], save_dir=None):
        self._gen_dir(save_dir)

        for res in res_data:
            test_case = TestCase.from_uuid(res.test_case_uuid)
            test_data = test_case.eval_data()
            for metric, errs in res.test_results.items():
                plt.figure()
                plt.title(f"{res.model_name} - {test_case.dataset_name} - {test_case.task_name}: ErrorGrade{{{metric}}}")
                err_list = sorted([(i, err) for i, err in enumerate(errs)], key=lambda x: x[1])
                if self._max_samples is None:
                    indices = list(range(len(test_data)))
                else:
                    indices = np.linspace(0, len(test_data) - 1, num=self._max_samples).astype(np.int32)

                for i, index in enumerate(indices):
                    in_tr, _ = test_data[err_list[index][0]]
                    c = i / (len(indices) - 1)
                    color = mpl.colors.to_hex((1-c) * np.array(mpl.colors.to_rgb("blue")) + c * np.array(mpl.colors.to_rgb("red")))
                    plt.gca().plot(in_tr[:, 0], in_tr[:, 1], "d-", color=color)
                    plt.gca().plot(in_tr[:1, 0], in_tr[:1, 1], "ko", markersize=10, markerfacecolor="none")

                lims = test_case.plot_limits
                plt.xlim(lims[0])
                plt.ylim(lims[1])

                if save_dir is None:
                    plt.show()
                else:
                    plt.savefig(f"{save_dir}/ErrorsColorGrade__{res.model_name}-{test_case.dataset_name}-{test_case.task_name}-{metric}.png", dpi=300, format="png", bbox_inches='tight')
                    plt.close()


class ErrorHistogram(VisOption):
    def __call__(self, res_data: List[ResData], save_dir=None):
        self._gen_dir(save_dir)

        for res in res_data:
            test_case = TestCase.from_uuid(res.test_case_uuid)
            test_data = test_case.eval_data()
            for metric, errs in res.test_results.items():
                plt.figure()
                plt.title(f"{res.model_name} - {test_case.dataset_name} - {test_case.task_name}: ErrorHistogram{{{metric}}}")

                plt.hist(errs, bins=len(errs)//10)

                if save_dir is None:
                    plt.show()
                else:
                    plt.savefig(f"{save_dir}/ErrorsColorGrade__{res.model_name}-{test_case.dataset_name}-{test_case.task_name}-{metric}.png", dpi=300, format="png", bbox_inches='tight')
                    plt.close()
