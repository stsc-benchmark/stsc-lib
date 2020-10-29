from enum import Enum


class Tasks(Enum):
    REPRESENTATION = 0
    DENOISING = 1
    NOISE_PREDICTION = 2  # target is noisy trajectory
    SMOOTH_PREDICTION = 3  # target is smooth trajectory (usually not given)
    # TODO: einen task bei man noise-noise trainiert und dann gegen noise-smooth vergleicht? weil modelle lernen ja ohnehin einen ausgleich


class LengthMode(Enum):
    FIXED = 0
    VARIABLE = 1


class ProcessingMode(Enum):
    FULL = 0
    SLIDING_WINDOW = 1
    RECURRENT = 2


class Target(Enum):
    TRAJECTORY = 0  # TODO: ADE
    DISTRIBUTION = 1  # TODO: lower/upper bound to kl-div, NLL
    SAMPLE_DISTRIBUTION = 2  # sample-based representation of a distribution -> TODO: unscented kl-div, sample-based NLL??
