# basically the same as example_prediction_task but using all the standard_eval constants and datasets (should be a little bit shorter then)
# -> possibly add some helper functions for loading stuff to make code even shorter
# -> make main parameterizable with different datasets (hyang, diamond, tmaze?)

# tests: 
# - multi-mode performance [nll, wasserstein, kl]
# - ml performance [ade] 
# - reaktionsfähigkeit [zeitschritte bis mode wechsel (collapse)] 
# - varianzschätzung [zero mean vergleich] -> evtl. ellipse vergleichen oder KL (für unimode datensatz und gauss_kde hat das closed form)
#
# -> achte darauf, welche datensätze man für was nimmt. varianzschätzung geht am besten auf unimode datensätzen

# man kann von bench helper klasse je task den datensatz, die obs/pred lens, train/test daten abfragen und so abfragen

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from stsc.baseline_models.red import REDPredictor2D
import stsc.evaluation.standard_evaluation as se
from stsc.evaluation.factors.evaluation_factor import EvaluationFactor, ResultsInfoPlotOptions
from stsc.datagen.sequence_sample_set import SequenceSampleSet
import stsc.misc.helpers as h
from stsc.misc.print_writer import PrintTarget, PrintWriter


# TODO: bei sample-basierten metriken auch distanz zu gt selbst (mit zusätzlichen samplesets) als baseline berechnen und angeben (zb bei mm performance)
# TODO: requirements: scipy==1.9.3 (1.10.1 hat einen bug in multivariate package) (oder req < 1.10 oder so)
# TODO: comments

if __name__ == "__main__":
    # we will be using the same parameterization of the multi-mode red predictor for all factors -> geht nicht wegen obs/pred len variations
    #red = REDPredictor2D(obs_len, 64, 1, pred_len, 3, device_str=torch_device)
    # red hyperparams
    hidden_size = 64
    num_hidden_layers = 1
    mixture_components = 5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    now = datetime.now()
    res_out_file = "out/red/out.txt"
    with open(res_out_file, "a") as f: 
        f.write(f"Run at {now.strftime('%d/%m/%Y %H:%M:%S')}\n\n")

    # use diamond dataset for variance estimation accuracy (TODO: need to add more datasets?)
    # use tmaze dataset for outlier robustness and responsiveness 
    # use diamond, tmaze and synth hyang datasets for overall prediction performance (TODO: add more biased datasets)
    # TODO: implement and add baseline models (GAN, VAE and R-MDN variants)
    for factor in se.StandardEvaluation.all_tests():  
        factor: EvaluationFactor  # type hint
        print(factor.name)

        # TODO: add possibility to run n-folds?
        for train_data_name, training_dataset in factor.iter_training_datasets():
            min_obs_ind = np.argmin(factor.observation_lengths(train_data_name))
            min_obs_len = factor.observation_lengths(train_data_name)[min_obs_ind]
            pred_len = factor.prediction_lengths(train_data_name)[min_obs_ind] 
            base_save_path = f"out/red_{train_data_name}"
            if not os.path.exists(f"{base_save_path}.pth"):
                red = REDPredictor2D(min_obs_len, hidden_size, num_hidden_layers, pred_len, mixture_components, device)
                red.estimate_parameters(training_dataset, 20, min_obs_len, pred_len, n_epochs=100, verbose=True)  # n_epochs = 100 
                red.save(base_save_path)
            else:
                red = REDPredictor2D.from_file(base_save_path, map_location=device)

            for test_dataset in factor.iter_test_datasets(train_data_name): 
                test_dataset_name = test_dataset.name
                test_dataset = np.array(test_dataset.data)
                pred_samples = {}
                for obs_len in factor.observation_lengths(train_data_name):
                    pred_samples[obs_len] = SequenceSampleSet(red.predict(test_dataset[:, obs_len-min_obs_len:obs_len], sample_pred=True))
                res = factor.evaluate_metrics(pred_samples,
                                            print_writer=PrintWriter((PrintTarget.FILE, res_out_file), del_file=False),
                                            results_info_plots_options=ResultsInfoPlotOptions("out/red", False, True, True))
                print(f"Testset '{test_dataset_name}': {res}")
            print()
                
        with open(res_out_file, "a") as f: 
            f.write("\n\n")
