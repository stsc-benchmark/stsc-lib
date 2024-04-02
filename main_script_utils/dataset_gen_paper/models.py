from stsc.benchmark.baseline_models.red import REDPredictor2D
from stsc.benchmark.baseline_models.vae import VAESeqPredictor2D
from stsc.benchmark.baseline_models.gan import GANSequencePredictor2D


def get_models(obs_len, pred_len, device: str = "cuda:0"):
    return (
        REDPredictor2D(obs_len, 64, 1, pred_len, 5, device),
        VAESeqPredictor2D(offset_mode=True, torch_device=device),
        GANSequencePredictor2D(torch_device=device)
    )