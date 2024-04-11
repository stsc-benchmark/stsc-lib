## Description [under construction]
This repository contains the latest version of the STSC benchmark and consists of the two primary packages [stsc.benchmark](/stsc/benchmark) and [stsc.datagen](/stsc/datagen). The datagen package provides a standalone pattern-based dataset generator for synthetic trajectory datasets, which provides ground truth probability distributions over full trajectories and is used in the benchmark package. 

## Requirements
- STSC requirements listed in requirements.txt: matplotlib, numpy, POT, scipy, seaborn
- Additional requirements: pytorch

## Publications and Technical Reports
[Generating Synthetic Ground Truth Distributions for Multi-step Trajectory Prediction using Probabilistic Composite Bézier Curves](https://arxiv.org/abs/2404.04397): This is a short paper about the mathematical foundation of the ground truth generation approach used in the benchmark and implemented (with some restrictions) in the [stsc.datagen](/stsc/datagen) package. 
