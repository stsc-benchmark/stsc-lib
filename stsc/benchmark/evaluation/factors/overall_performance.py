from __future__ import annotations
from typing import List, Dict, Optional
import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import stsc.datagen.predefined_datasets as predata
from stsc.datagen.sequence_sample_set import SequenceSampleSet
from stsc.benchmark.misc.print_writer import PrintWriter
from stsc.benchmark.evaluation.factors.evaluation_factor import EvaluationFactor, ResultsInfoPlotOptions, TestDataset
import stsc.benchmark.evaluation.standard_evaluation_constants as consts
from stsc.benchmark.evaluation.metrics import Metrics
import stsc.benchmark.evaluation.metrics as metr
from stsc.benchmark.misc.overrides import overrides
import stsc.benchmark.misc.seq_plot as seq_plot


class OverallPerformance(EvaluationFactor):
    """
    Measures the overall prediction performance of a trajectory prediction model in a multi-modal setting.

    Dataset: Synthetic Hyang
    - Simplified synthetic replica of the hyang scene in the stanford drone dataset.
    -- Data speed distribution is unimodal
    -- Uniform path distribution

    Performance Measures: 
    - NLL: Common performance measure calculating the negative log-likelihood using the predicted distribution and given test trajectory samples.
    - KL-Divergence: As we have the actual ground truth distribution available, the KL-Divergence between a predicted and the ground truth distribution can be calculated, allowing for a more accurate comparison.
    - Wasserstein Distance: Another metric for comparing probability distributions.
    - ADE: Average Displacement Error calculated from a maximum likelihood estimate. For completeness
    """
    def __init__(self) -> None:
        EvaluationFactor.__init__(self, "Test factor: Overall performance")
        self._init_datasets()
        self._init_gt()

    def _init_datasets(self) -> None:
        data_file_path = os.path.join(self._stsc_dir, "files/datasets/overall_performance.pkl")
        if not os.path.exists(data_file_path):
            # sample train/test datasets with an 80/20 split
            test_seq_lens = [8, 10, 15]  # test sequences length per dataset
            for i, dataset in enumerate([predata.diamond(), predata.balanced_tmaze_longtail(), predata.synth_hyang()]):
                n_train_samples = len(dataset.weights) * 50  # use <number_of_paths> * 50 training samples
                n_test_samples = int(n_train_samples / 0.8 * 0.2)  # "80/20" split, but test dataset is inflated, as samples will be sliced into overlapping pieces of given sequence length (samples from different components in dataset can have different lengths)
                test_seq_len = test_seq_lens[i]
                self._data_gens[dataset.name] = dataset
                self._training_datasets[dataset.name] = dataset.sample(n_train_samples)
                # prepare test dataset
                samples = dataset.sample(n_test_samples) 
                test_data = []
                start_indices = []
                for s in samples:
                    for j in range(0, len(s) + 1 - test_seq_len, 5):  # TODO: 5 step?
                        test_data.append(s[j:j+test_seq_len])
                        start_indices.append(j)
                self._test_datasets[dataset.name] = [TestDataset(name="Test", data=np.asarray(test_data), start_indices=start_indices)]
            self._dump_datasets(data_file_path)
        else:
            self._load_datasets(data_file_path)

    def _init_gt(self) -> None:  
        # calculate gt distribution sample sets for each dataset
        data_file_path_base = os.path.join(self._stsc_dir, "files/datasets/overall_performance_gt")
        if not os.path.exists(f"{data_file_path_base}_samples.pkl"):
            self._test_sample_seqs = {}
            self._test_posteriors = {}
            for ds in self._data_gens.keys():
                obs_len = self.observation_lengths(ds)[0]
                pred_len = self.prediction_lengths(ds)[0]
                data_gen = self._data_gens[ds]
                test_dataset = self._test_datasets[ds][0]
                self._test_posteriors[ds] = []
                samples = np.empty(shape=[len(test_dataset.data), pred_len, consts.n_gt_samples, 2])
                for i, test_seq in enumerate(test_dataset.data):  
                    cond_inds = list(np.arange(obs_len).astype(int) + test_dataset.start_indices[i])
                    data_gen.posterior(test_seq[:obs_len], cond_inds)
                    gmm_seq = data_gen.sequence_distribution()
                    self._test_posteriors[ds].append(gmm_seq)
                    for step in range(pred_len):
                        samples[i, step] = gmm_seq(step).sample(n_samples=consts.n_gt_samples)
                self._test_sample_seqs[ds] = SequenceSampleSet(samples)

            with open(f"{data_file_path_base}_samples.pkl", "wb") as f:
                pkl.dump({ds: sss.samples for ds, sss in self._test_sample_seqs.items()}, f)
            with open(f"{data_file_path_base}_posteriors.pkl", "wb") as f:
                pkl.dump(self._test_posteriors, f)
            data_gen.prior()
        else:
            with open(f"{data_file_path_base}_samples.pkl", "rb") as f:
                data = pkl.load(f)
                self._test_sample_seqs = {ds: SequenceSampleSet(samples) for ds, samples in data.items()}
            with open(f"{data_file_path_base}_posteriors.pkl", "rb") as f:
                self._test_posteriors = pkl.load(f)

    @overrides(EvaluationFactor) 
    def observation_lengths(self, training_dataset_name: str) -> List[int]:
        if training_dataset_name == "diamond":
            return [2]
        elif training_dataset_name == "tmaze_longtail":
            return [4]
        elif training_dataset_name == "synth_hyang":
            return [5]
        else:
            raise Exception(f"Unsupported dataset '{training_dataset_name}'.")

    @overrides(EvaluationFactor)
    def prediction_lengths(self, training_dataset_name: str) -> List[int]:
        if training_dataset_name == "diamond":
            return [6]
        elif training_dataset_name == "tmaze_longtail":
            return [6]
        elif training_dataset_name == "synth_hyang":
            return [10]
        else:
            raise Exception(f"Unsupported dataset '{training_dataset_name}'.")

    @overrides(EvaluationFactor)
    def evaluate_metrics(
        self, 
        prediction_samples: Dict[int,SequenceSampleSet], 
        print_writer: Optional[PrintWriter] = None, 
        results_info_plots_options: Optional[ResultsInfoPlotOptions] = None, 
        model_name: Optional[str] = None
    ) -> Dict[int, Dict[Metrics,float]]: 
        i = self._cur_test_iter_index  # always 0 as there is only 1 test dataset for each train dataset
        ds = self._cur_train_iter_key
        pw: PrintWriter = self._setup_print_writer(print_writer)
        self._setup_plot_dir(results_info_plots_options)
        plot_model_prefix = model_name + "_" if model_name is not None else ""
        res_dict = {}

        pw.write_line(f":: Overall Performance [Dataset: {ds}] ::")
        if model_name is not None:
            pw.write_line(f"Evaluating model: {model_name}")

        obs_len = self.observation_lengths(ds)[0]
        pred_len = self.prediction_lengths(ds)[0]
        test_dataset = self._test_datasets[ds][0]
        test_posteriors = self._test_posteriors[ds]
        pred_samples = prediction_samples[obs_len]

        ade = metr.ade(pred_samples, test_dataset.data[:, obs_len:], ret_mean=False)
        nll = metr.nll(pred_samples, test_dataset.data[:, obs_len:], ret_mean=False)
        kl = metr.kl_div(pred_samples, test_posteriors, test_dataset.data[:, :obs_len], ret_mean=False)
        wasserstein = metr.wasserstein(pred_samples, test_posteriors, test_dataset.data[:, :obs_len], ret_mean=False, n_seeds=5, n_projections=100)  

        res_dict[obs_len] = {
            Metrics.ADE: np.mean(ade),
            Metrics.NLL: np.mean(nll),
            Metrics.KL_DIVERGENCE: np.mean(kl),
            Metrics.WASSERSTEIN_DISTANCE: np.mean(wasserstein)
        }

        pw.buffer_line(f"Observation length: {obs_len}")
        pw.buffer_line(f"  ADE: {res_dict[obs_len][Metrics.ADE]}")
        pw.buffer_line(f"  NLL: {res_dict[obs_len][Metrics.NLL]}")
        pw.buffer_line(f"  KL: {res_dict[obs_len][Metrics.KL_DIVERGENCE]}")
        pw.buffer_line(f"  Wasserstein: {res_dict[obs_len][Metrics.WASSERSTEIN_DISTANCE]}")
        pw.buffer_line("")
        pw.flush()

        # TODO: get rid of duplicate colors
        if results_info_plots_options is not None:
            # ade
            plot_inds = [np.argmin(ade), np.argmax(ade)]
            err_vals = [np.min(ade), np.max(ade)]
            err_names = ["ADE (best)", "ADE (worst)"]
            for ii in range(len(plot_inds)):
                inp_sample = self._test_datasets[ds][0].data[plot_inds[ii]]
                plt.figure()
                plt.plot(inp_sample[:, 0], inp_sample[:, 1], "ko--")
                plt.plot(inp_sample[:obs_len, 0], inp_sample[:obs_len, 1], "go")
                mseq = []
                for j in range(pred_len):
                    p = prediction_samples[obs_len].step_samples(plot_inds[ii], j)
                    mseq.append(np.mean(p, axis=0))
                mseq = np.asarray(mseq)
                plt.plot(mseq[:, 0], mseq[:, 1], "rs-", markerfacecolor="none", label="pred")
                if results_info_plots_options.show_titles:
                    plt.title(f"{err_names[ii]}: {err_vals[ii]}")
                plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}overall_performance_{ds}_ade_{'best' if ii == 0 else 'worst'}.png", dpi=300, bbox_inches="tight")
                plt.close()

            # nll
            plot_inds = [np.argmin(nll), np.argmax(nll)]
            err_vals = [np.min(nll), np.max(nll)]
            err_names = ["NLL (best)", "NLL (worst)"]
            for ii in range(len(plot_inds)):
                inp_sample = self._test_datasets[ds][0].data[plot_inds[ii]]
                plt.figure()
                for j in range(pred_len):
                    p = prediction_samples[obs_len].step_samples(plot_inds[ii], j)
                    plt.plot(p[:, 0], p[:, 1], "rs", markerfacecolor="none", alpha=0.5, label="pred")
                plt.plot(inp_sample[:, 0], inp_sample[:, 1], "ko--")
                plt.plot(inp_sample[:obs_len, 0], inp_sample[:obs_len, 1], "go")
                if results_info_plots_options.show_titles:
                    plt.title(f"{err_names[ii]}: {err_vals[ii]}")
                plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}overall_performance_{ds}_nll_{'best' if ii == 0 else 'worst'}.png", dpi=300, bbox_inches="tight")
                plt.close()

            # kl, wasserstein
            plot_inds = [
                np.argmin(kl), np.argmax(kl),
                np.argmin(wasserstein), np.argmax(wasserstein)
            ]
            err_vals = [
                np.min(kl), np.max(kl),
                np.min(wasserstein), np.max(wasserstein)
            ]
            err_names = ["KL (best)", "KL (worst)", "Wasserstein (best)", "Wasserstein (worst)"]
            for ii in range(len(plot_inds)):
                inp_sample = self._test_datasets[ds][0].data[plot_inds[ii]]
                plt.figure()
                plt.plot(inp_sample[:, 0], inp_sample[:, 1], "ko--")
                plt.plot(inp_sample[:obs_len, 0], inp_sample[:obs_len, 1], "go")
                for j in range(pred_len):
                    p = prediction_samples[obs_len].step_samples(plot_inds[ii], j)
                    plt.plot(p[:, 0], p[:, 1], "rs", markerfacecolor="none", alpha=0.5)
                posterior = test_posteriors[plot_inds[ii]]
                pmeans = [ms[:pred_len] for ms in posterior.mean_vectors]
                pcovs = [cs[:pred_len] for cs in posterior.covar_matrices]
                seq_plot.plot_gaussian_sequence_mixture(posterior.weights, pmeans, pcovs, suppress_alpha=True, show=False)
                if results_info_plots_options.show_titles:
                    plt.title(f"{err_names[ii]}: {err_vals[ii]}")
                plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}overall_performance_{ds}_{err_names[ii].split()[0].lower()}_{'best' if ii % 2 == 0 else 'worst'}.png", dpi=300, bbox_inches="tight")
                plt.close()

            # kl/wasserstein best/worst per step
            plot_inds = [np.argmin(kl), np.argmax(kl), np.argmin(wasserstein), np.argmax(wasserstein)]
            err_vals = [np.min(kl), np.max(kl), np.min(wasserstein), np.max(wasserstein)]
            err_names = ["KL (best)", "KL (worst)", "Wasserstein (best)", "Wasserstein (worst)"]
            err_prefix = ["kl", "kl", "wasserstein", "wasserstein"]
            for ii in range(len(plot_inds)):
                inp_sample = self._test_datasets[ds][0].data[plot_inds[ii]]
                posterior = test_posteriors[plot_inds[ii]]
                pmeans = [ms[:pred_len] for ms in posterior.mean_vectors]
                pcovs = [cs[:pred_len] for cs in posterior.covar_matrices]
                for j in range(pred_len):
                    plt.figure()
                    p = prediction_samples[obs_len].step_samples(plot_inds[ii], j)
                    plt.plot(p[:, 0], p[:, 1], "rs", markerfacecolor="none", alpha=0.5, label="pred")
                    sns.kdeplot(x=p[:, 0], y=p[:, 1], color="r")  
                    seq_plot.plot_gaussian_sequence_mixture(posterior.weights, [m[j:j+1] for m in pmeans], [c[j:j+1] for c in pcovs], suppress_alpha=True, show=False)
                    plt.plot(inp_sample[:, 0], inp_sample[:, 1], "ko--")
                    plt.plot(inp_sample[:obs_len, 0], inp_sample[:obs_len, 1], "go")
                    if results_info_plots_options.show_titles:
                        plt.title(f"{err_names[ii]} pred step {j}")
                    plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}overall_performance_{ds}_{err_prefix[ii]}_{'best' if ii == 0 else 'worst'}_step-{str(j).zfill(3)}.png", dpi=300, bbox_inches="tight")
                    plt.close()
       
            if results_info_plots_options.plot_all_test:
                # samples
                for ii in range(len(self._test_datasets[ds][0].data)):
                    inp_sample = self._test_datasets[ds][0].data[ii] 
                    plt.figure()
                    plt.plot(inp_sample[:, 0], inp_sample[:, 1], "ko--")
                    plt.plot(inp_sample[:obs_len, 0], inp_sample[:obs_len, 1], "go")
                    for j in range(pred_len):
                        p = prediction_samples[obs_len].step_samples(ii, j)
                        plt.plot(p[:, 0], p[:, 1], "rs", markerfacecolor="none", alpha=0.5)
                    posterior = test_posteriors[ii]
                    pmeans = [ms[:pred_len] for ms in posterior.mean_vectors]
                    pcovs = [cs[:pred_len] for cs in posterior.covar_matrices]
                    seq_plot.plot_gaussian_sequence_mixture(posterior.weights, pmeans, pcovs, suppress_alpha=True, show=False)
                    if results_info_plots_options.show_titles:
                        plt.title(f"Test Sample Index: {ii}")
                    plt.savefig(f"{results_info_plots_options.directory_path}/{plot_model_prefix}overall_performance_{ds}_index-{str(ii).zfill(3)}.png", dpi=300, bbox_inches="tight")
                    plt.close()

        return res_dict
