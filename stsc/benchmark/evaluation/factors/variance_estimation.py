from __future__ import annotations
from typing import List, Dict, Optional
import os
import pickle as pkl
import functools

import numpy as np
import matplotlib.pyplot as plt

import stsc.datagen.predefined_datasets as predata
from stsc.datagen.sequence_sample_set import SequenceSampleSet
import stsc.benchmark.evaluation.metrics as metr
import stsc.benchmark.evaluation.standard_evaluation_constants as consts
from stsc.benchmark.misc.print_writer import PrintWriter
from stsc.benchmark.evaluation.metrics import Metrics
from stsc.benchmark.evaluation.factors.evaluation_factor import EvaluationFactor, ResultsInfoPlotOptions, TestDataset
from stsc.benchmark.misc.overrides import overrides


class VarianceEstimation(EvaluationFactor):
    """
    Gives an estimate on how well a prediction model is capable of predicting the actual variance in the underlying data distribution on a per time-step basis.

    Dataset: Diamond 
    - Combines different velocity-related motion patterns present in common trajectory prediction datasets, i.e. constant slow and fast forward motion (constant velocity), acceleration and deceleration.
    - Each pattern moves in a different direction in order to provide the prediction model with a directional bias for the prediction.

    Performance Measure: KL-Divergence
    - Test sets are build in a way that there is always only 1 valid mode in the distribution to be predicted. This is done in order to reduce the possibility for harder to measure and/or control side effects. Further, the target distribution becomes Gaussian, allowing for a closed-form solution for KL-Divergence.
    - We provide a reference baseline value through a "self-KL" value, which calculates the KL-Divergence between multiple sample sets taken from the ground truth distribution.
    """
    def __init__(self) -> None:
        EvaluationFactor.__init__(self, "Test factor: Variance Estimation")
        self._init_datasets()
        self._init_gt()

    def _init_datasets(self) -> None:
        data_file_path = os.path.join(self._stsc_dir, "files/datasets/var_estimation.pkl")
        if not os.path.exists(data_file_path):
            dg = predata.diamond()
            self._data_gens[dg.name] = dg
            self._training_datasets[dg.name] = functools.reduce(lambda x,y: x+y, [dg.sample_component(k, 100) for k in range(dg.n_comps)])
            tnames = ["Slow", "Fast", "Acceleration", "Deceleration"]
            self._test_datasets[dg.name] = [TestDataset(name=tnames[k], data=dg.sample_component(k, 20)) for k in range(dg.n_comps)]

            self._dump_datasets(data_file_path)
        else:
            self._load_datasets(data_file_path)

    def _init_gt(self) -> None:
        data_file_path = os.path.join(self._stsc_dir, "files/datasets/var_estimation_gt_samples.pkl")
        if not os.path.exists(data_file_path):
            tdname = list(self._data_gens.keys())[0]  # training dataset name
            data_gen = self._data_gens[tdname]
            self._test_sample_seqs = []
            for test_dataset in self._test_datasets[tdname]:
                self._test_sample_seqs.append([])
                # Provide 6 sets of samples per test dataset. 
                # 1 for actual comparison and 5 for providing a baseline.
                for _ in range(6):
                    #[ n_sequences, sequence_length, n_samples, 2 ]
                    s = np.empty(shape=[len(test_dataset.data), 6, consts.n_gt_samples, 2])
                    for i, test_seq in enumerate(test_dataset.data):
                        data_gen.posterior(test_seq[:2], [0, 1])
                        gmm_seq = data_gen.sequence_distribution()
                        for step in range(len(test_seq[2:])):
                            s[i, step] = gmm_seq(step).sample(n_samples=consts.n_gt_samples)
                    self._test_sample_seqs[-1].append(SequenceSampleSet(s))
            with open(data_file_path, "wb") as f:
                pkl.dump([[sss.samples for sss in sseqs] for sseqs in self._test_sample_seqs], f)
            data_gen.prior()
        else:
            with open(data_file_path, "rb") as f:
                data = pkl.load(f)
                self._test_sample_seqs = [[SequenceSampleSet(samples) for samples in e] for e in data]

    @overrides(EvaluationFactor) 
    def observation_lengths(self, *_unused_args, **_unused_kwargs) -> List[int]:
        del _unused_args, _unused_kwargs
        return [2]

    @overrides(EvaluationFactor)
    def prediction_lengths(self, *_unused_args, **_unused_kwargs) -> List[int]:
        del _unused_args, _unused_kwargs
        return [6]

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

        pw.write_line(f":: Variance Estimation [Dataset: {tkey}] ::")
        if model_name is not None:
            pw.write_line(f"Evaluating model: {model_name}")

        obs_len = self.observation_lengths(tkey)[0]
        kl_vals = metr.seq_gauss_kl_div(self._test_sample_seqs[i][0], prediction_samples[obs_len], zero_mean=True)
        self_kls = [np.mean(metr.seq_gauss_kl_div(self._test_sample_seqs[i][0], self._test_sample_seqs[i][set_i], zero_mean=True)) for set_i in range(1, 6)]
        kl = np.mean(kl_vals)
        res_dict[Metrics.KL_DIVERGENCE] = kl

        pw.buffer_line(f"Test dataset: {self._test_datasets[tkey][i].name}")
        pw.buffer_line(f"  Gaussian KL: {kl}")
        pw.buffer_line(f"  Self-KL (5 sample sets): {np.mean(self_kls)} +- {np.std(self_kls)}")
        pw.buffer_line("")
        pw.flush()

        if results_info_plots_options is not None:
            # best/worst
            pred_len = self.prediction_lengths(tkey)[0]
            file_name_base = f"variance_estimation_{self._test_datasets[tkey][i].name.lower().replace(' ', '')}"
            inds = [np.argmin(kl_vals), np.argmax(kl_vals)]
            err_vals = [np.min(kl_vals), np.max(kl_vals)]
            err_names = ["KL (best)", "KL (worst)"]
            for ii in range(len(inds)):
                inp_sample = self._test_datasets[tkey][i].data[inds[ii]]
                
                for j in range(pred_len):
                    plt.figure()
                    p = prediction_samples[obs_len].step_samples(inds[ii], j)
                    p = p - np.mean(p, axis=0)
                    gt = self._test_sample_seqs[i][0].step_samples(inds[ii], j) 
                    gt = gt - np.mean(gt, axis=0)
                    plt.plot(p[:, 0], p[:, 1], "rs", markerfacecolor="none", alpha=0.5, label="pred")
                    plt.plot(gt[:, 0], gt[:, 1], "gd", markerfacecolor="none", alpha=0.5, label="gt")
                    if results_info_plots_options.show_titles:
                        plt.title(f"{err_names[ii]}: {err_vals[ii]}")
                    plt.legend()
                    plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}{file_name_base}_kl_{'best' if ii == 0 else 'worst'}_step-{str(j)}.png", dpi=300, bbox_inches="tight")
                    plt.close()
        
        return res_dict
