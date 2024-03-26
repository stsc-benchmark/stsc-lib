from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional
import os
import pickle as pkl
import functools

import numpy as np
import matplotlib.pyplot as plt

import stsc.datagen.predefined_datasets as predata
from stsc.datagen.sequence_sample_set import SequenceSampleSet
import stsc.benchmark.evaluation.metrics as metr
from stsc.benchmark.evaluation.metrics import Metrics
from stsc.benchmark.misc.print_writer import PrintWriter
import stsc.benchmark.evaluation.standard_evaluation_constants as consts
from stsc.benchmark.evaluation.factors.evaluation_factor import EvaluationFactor, ResultsInfoPlotOptions, TestDataset
from stsc.benchmark.misc.overrides import overrides


class OutlierRobustness(EvaluationFactor):
    """
    Examines the robustness of a prediction model under the presence of single outliers in the observation. Throughout the test data, the position of the outliers moves from the beginning to the end of the observation, as prediction models oftentimes rely heavily on the most recent observations.

    Dataset: T-Maze
    - Resembles a t-junction.
    - Given observations in the test datasets end prior to the junction, thus enforcing a bi-modal ground truth distribution for future time steps.

    Performance Measure: KL-Divergence
    - Calculates a sample-based KL-Divergence between the predicted multi-modal distribution and the ground truth distribution.
    """
    def __init__(self) -> None:
        EvaluationFactor.__init__(self, "Test factor: Robustness to outliers")
        self._init_datasets()
        self._init_gt()

    def _init_datasets(self) -> None:
        data_file_path = os.path.join(self._stsc_dir, "files/datasets/outlier_robustness.pkl")
        if not os.path.exists(data_file_path):
            dg = predata.balanced_tmaze()
            self._data_gens[dg.name] = dg
            self._training_datasets[dg.name] = functools.reduce(lambda x,y: x+y, [dg.sample_component(k, 100, cap_length=15) for k in range(dg.n_comps)])
            # provide 4 test sets: 
            # - 1 without outliers 
            self._test_datasets[dg.name] = [TestDataset(name="No Outliers", data=np.asarray(functools.reduce(lambda x,y: x+y, [dg.sample_component(k, 20, cap_length=15) for k in range(dg.n_comps)])))]
            # - 4 with outliers at the 2nd, 3rd, 4th and 5th (last point) observed point (20 each), respectively.
            # -- re-use the sampled sequences without outliers and add those in
            for outlier_pos in [1, 2, 3, 4]:
                samples = []
                for i, sample in enumerate(self._test_datasets[dg.name][0].data):
                    outlier_sample = np.copy(sample)
                    outlier_sample[outlier_pos, 0] = 1.5 if i % 2 == 0 else -1.5 # set x value to +/- 1.5
                    samples.append(outlier_sample)
                self._test_datasets[dg.name].append(TestDataset(name=f"Outlier at position {outlier_pos+1}", data=np.asarray(samples)))
            
            self._dump_datasets(data_file_path)
        else:
            self._load_datasets(data_file_path)

    def _init_gt(self) -> None: 
        # use the same sample set sequences for all test datasets (gt target distribution does not consider outliers)
        data_file_path_base = os.path.join(self._stsc_dir, "files/datasets/outlier_robustness_gt")
        if not os.path.exists(f"{data_file_path_base}_samples.pkl"):
            tdname = list(self._data_gens.keys())[0]  # training dataset name
            data_gen = self._data_gens[tdname]
            obs_len = self.observation_lengths(tdname)[0]
            pred_len = self.prediction_lengths(tdname)[0]
            self._test_posteriors = []
            self._test_sample_seqs = []
            s = np.empty(shape=[len(self._test_datasets[tdname][0].data), pred_len, consts.n_gt_samples, 2])
            for i, test_seq in enumerate(self._test_datasets[tdname][0].data):
                data_gen.posterior(test_seq[:obs_len], list(range(obs_len)))
                gmm_seq = data_gen.sequence_distribution()
                self._test_posteriors.append(gmm_seq)
                for step in range(pred_len):
                    s[i, step] = gmm_seq(step).sample(n_samples=consts.n_gt_samples)

            for _ in range(len(self._test_datasets[tdname])):
                self._test_sample_seqs.append(SequenceSampleSet(s))

            with open(f"{data_file_path_base}_samples.pkl", "wb") as f:
                pkl.dump([sss.samples for sss in self._test_sample_seqs], f)
            with open(f"{data_file_path_base}_posteriors.pkl", "wb") as f:
                pkl.dump(self._test_posteriors, f)
            data_gen.prior()
        else:
            with open(f"{data_file_path_base}_samples.pkl", "rb") as f:
                data = pkl.load(f)
                self._test_sample_seqs = [SequenceSampleSet(samples) for samples in data]
            with open(f"{data_file_path_base}_posteriors.pkl", "rb") as f:
                self._test_posteriors = pkl.load(f)

    @overrides(EvaluationFactor) 
    def observation_lengths(self, *_unused_args, **_unused_kwargs) -> List[int]:
        del _unused_args, _unused_kwargs
        return [5]

    @overrides(EvaluationFactor)
    def prediction_lengths(self, *_unused_args, **_unused_kwargs) -> List[int]:
        del _unused_args, _unused_kwargs
        return [10]
    
    @overrides(EvaluationFactor) 
    def evaluate_metrics(
        self, 
        prediction_samples: Dict[int,SequenceSampleSet], 
        print_writer: Optional[PrintWriter] = None, 
        results_info_plots_options: Optional[ResultsInfoPlotOptions] = None, 
        model_name: Optional[str] = None
    ) -> Dict[int, Dict[Metrics,float]]: 
        i = self._cur_test_iter_index
        tkey = self._cur_train_iter_key
        pw: PrintWriter = self._setup_print_writer(print_writer)
        self._setup_plot_dir(results_info_plots_options)
        plot_model_prefix = model_name + "_" if model_name is not None else ""
        res_dict = {}

        pw.write_line(f":: Outlier Robustness [Dataset: {tkey}] ::")
        if model_name is not None:
            pw.write_line(f"Evaluating model: {model_name}")

        obs_len = self.observation_lengths(tkey)[0]
        kl_vals = metr.kl_div(prediction_samples[obs_len], self._test_posteriors, self._test_datasets[tkey][i].data[:, :obs_len], ret_mean=False, n_processes=1)
        kl = np.mean(kl_vals)
        res_dict[Metrics.KL_DIVERGENCE] = kl

        pw.buffer_line(f"Test dataset: {self._test_datasets[tkey][i].name}")
        pw.buffer_line(f"  KL: {kl}")
        pw.buffer_line("")
        pw.flush()

        if results_info_plots_options is not None:
            # best/worst
            pred_len = self.prediction_lengths(tkey)[0]
            file_name_base = f"outlier_robustness_{self._test_datasets[tkey][i].name.lower().replace(' ', '')}"
            inds = [np.argmin(kl_vals), np.argmax(kl_vals)]
            err_vals = [np.min(kl_vals), np.max(kl_vals)]
            err_names = ["KL (best)", "KL (worst)"]
            for ii in range(len(inds)):
                inp_sample = self._test_datasets[tkey][i].data[inds[ii]]
                plt.figure()
                for j in range(pred_len):
                    p = prediction_samples[obs_len].step_samples(inds[ii], j)
                    plt.plot(p[:, 0], p[:, 1], "rs", markerfacecolor="none", alpha=0.5, label="pred")
                plt.plot(inp_sample[:, 0], inp_sample[:, 1], "ko--")
                plt.plot(inp_sample[:obs_len, 0], inp_sample[:obs_len, 1], "go")
                if results_info_plots_options.show_titles:
                    plt.title(f"{err_names[ii]}: {err_vals[ii]}")
                plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}{file_name_base}_kl_{'best' if ii == 0 else 'worst'}.png", dpi=300, bbox_inches="tight")
                plt.close()

        return res_dict

