from __future__ import annotations
from typing import List, Optional, Union, Tuple
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as tfun

from stsc.datagen.gmm_sequence import GMM
from stsc.benchmark.misc.mixin import BatchGenMixin


class REDPredictor2D(torch.nn.Module, BatchGenMixin):
    """
    Implements a variant of the RED predictor [1] for multi-modal predictions.

    [1] Becker, S., Hug, R., Hubner, W., & Arens, M. (2018). Red: A simple but effective baseline predictor for the trajnet benchmark. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops.
    """
    def __init__(
            self, 
            in_seq_len, 
            hidden_size, 
            num_hidden_layers, 
            out_seq_len, 
            mixture_components, 
            device_str="cuda:0"
        ) -> None:
        assert num_hidden_layers > 0, "<num_hidden_layers> must be at least 1."

        super(REDPredictor2D, self).__init__()

        self.in_seq_len = in_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.out_seq_len = out_seq_len
        self.mixture_components = mixture_components
        self.device = torch.device(device_str)

        self.__mask_std1 = torch.from_numpy(np.diag([1, 0])).float().to(self.device)
        self.__mask_std2 = torch.from_numpy(np.diag([0, 1])).float().to(self.device)
        self.__mask_corr = torch.from_numpy(np.reshape([0, 1, 1, 0], [2, 2])).float().to(self.device)
        
        self.lstm = torch.nn.LSTM(input_size=2,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_hidden_layers,
                                  bias=True,
                                  batch_first=True).to(self.device)
        self.mdn = torch.nn.Linear(hidden_size, self.mixture_components + self.out_seq_len * self.mixture_components * 5).to(self.device)
        self._val_ranges = torch.from_numpy(np.cumsum([
            0, 
            self.mixture_components,  # weights
            self.out_seq_len * self.mixture_components * 2,  # means
            self.out_seq_len * self.mixture_components * 2,  # stds
            self.out_seq_len * self.mixture_components  # corrs
        ])).int().to(self.device)

        self.__torch_pi = torch.tensor(np.pi).float().to(self.device)

    def save(self, base_path) -> None:
        torch.save(self.state_dict(), f"{base_path}.pth")
        with open(f"{base_path}.meta", "wb") as f:
            pkl.dump({
                "in_seq_len": self.in_seq_len,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "out_seq_len": self.out_seq_len,
                "mixture_components": self.mixture_components,
                "device_str": self.device
            }, f)

    @classmethod
    def from_file(cls, base_path, map_location=None) -> REDPredictor2D:
        with open(f"{base_path}.meta", "rb") as f:
            kwargs_dict = pkl.load(f)
        instance = cls(**kwargs_dict)
        
        if map_location is None:
            map_location = instance.device        
        instance.load_state_dict(torch.load(f"{base_path}.pth", map_location=map_location))

        return instance

    def predict(
            self, 
            inputs: np.ndarray, 
            sample_pred: bool = False, 
            n_step_samples: int = 100
        ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """ 
        Calculates predictions of the model given (a batch of) input sequences. 
        Returns the prediction either as a mixture of Gaussians, where each component corresponds to a single path with its modeled trajectory points flattened into a single vector/matrix, or a sequences of sample sets representing the respective distribution at each modeled time step.
        """
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, 0)

        assert len(inputs.shape) == 3, "<inputs> must have shape [ batch_size, sequence_length, 2 ]."
        assert inputs.shape[1] == self.in_seq_len, f"<inputs>' sequence length dimension ({inputs.shape[1]}) must match <self.in_seq_len> ({self.in_seq_len})."

        inputs_tensor = torch.from_numpy(inputs).float().to(self.device)
        pred_pi, pred_mu, pred_cov = self.forward(inputs_tensor, self.zero_state(inputs.shape[0]))

        pred_pi = pred_pi.data.cpu().numpy()
        pred_mu = pred_mu.data.cpu().numpy()
        pred_cov = pred_cov.data.cpu().numpy()

        if not sample_pred:
            return pred_pi, pred_mu, pred_cov
        
        samples = []
        for b in range(pred_pi.shape[0]):
            samples.append([])
            for t in range(pred_mu.shape[2]):
                step_samples = GMM(pred_pi[b], pred_mu[b, :, t], pred_cov[b, :, t]).sample(n_step_samples)
                samples[-1].append(step_samples)

        return np.array(samples)

    def forward(
            self, 
            inputs: torch.Tensor, 
            state: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes given input sequence into a single vector and maps it into the parameters of a GMM.
        <inputs> shape is expected to be [ batch_size, sequence_length, 2 ].
        
        Note: Input tensor location is not checked and thus no device transfers are triggered if the location does not match the model's location.
        """
        if state is None:
            state = self.zero_state(inputs.shape[0])

        batch_size = inputs.shape[0]
        lstm_out, _ = self.lstm(inputs, state)  # [ batch_size, in_seq_len, hidden_size ]
        inp_enc = lstm_out[:, -1]  # [ batch_size, hidden_size ]
        mdn_out = self.mdn(inp_enc)

        # This is a multi-mode version of RED.
        # Here, each component corresponds to a distinct path and each weight in the mixture weight set corresponds to one path.
        pi = torch.reshape(tfun.softmax(mdn_out[:, :self._val_ranges[1]], dim=1), [batch_size, self.mixture_components])
        mu_vals = torch.reshape(mdn_out[:, self._val_ranges[1]:self._val_ranges[2]], [batch_size, self.mixture_components, self.out_seq_len, 2])
        std_vals = torch.reshape(mdn_out[:, self._val_ranges[2]:self._val_ranges[3]], [batch_size, self.mixture_components, self.out_seq_len, 2])
        std_vals = tfun.softplus(std_vals)        
        corr_vals = torch.reshape(mdn_out[:, self._val_ranges[3]:], [batch_size, self.mixture_components, self.out_seq_len])
        # hack for removing numerical instabilities (corr = +- 1 seems to lead to non-invertible matrix due to high matrix condition)
        # TODO: this hack is a relic from original tensorflow 1.x code. Check if still relevant.
        corr_vals = torch.clamp(tfun.softsign(corr_vals), -0.99, 0.99)

        rep_ref = torch.tile(torch.reshape(inputs[:, -1], [batch_size, 1, 1, 2]), [1, self.mixture_components, 1, 1])
        mu = torch.cumsum(torch.cat([rep_ref, mu_vals], dim=2), dim=2)[:, :, 1:]

        std1 = torch.tile(torch.reshape(std_vals[:, :, :, 0], [batch_size, self.mixture_components, self.out_seq_len, 1, 1]), [1, 1, 1, 2, 2])
        std2 = torch.tile(torch.reshape(std_vals[:, :, :, 1], [batch_size, self.mixture_components, self.out_seq_len, 1, 1]), [1, 1, 1, 2, 2])
        corr_vals = torch.tile(torch.reshape(corr_vals, [batch_size, self.mixture_components, self.out_seq_len, 1, 1]), [1, 1, 1, 2, 2])
        cov = self.__mask_std1 * std1 * std1 + self.__mask_std2 * std2 * std2 + self.__mask_corr * std1 * std2 * corr_vals

        return pi, mu, cov

    def zero_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: although batch_first is set to True, the hidden state is still expected to be [ layers, bs, dim ]
        return (torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size]).to(self.device),  # h
                torch.zeros([self.num_hidden_layers, batch_size, self.hidden_size]).to(self.device))  # c

    def estimate_parameters(
            self, 
            data: List[np.ndarray], 
            batch_size: int,
            obs_len: int, 
            pred_len: int, 
            learning_rate: float = 1e-3, 
            n_epochs: int = 100,
            verbose: bool = True
        ) -> None:
        """ Estimate the model parameters using the Adam optimizer. """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        batch_gen = self.build_batch_generator(data, batch_size, obs_len + pred_len)
        epoch = 0
        while epoch < n_epochs:
            batch_data, inc_epoch = next(batch_gen)
            in_data = batch_data[:, :obs_len]
            target_data = batch_data[:, obs_len:]

            pred_pi, pred_mu, pred_cov = self.forward(in_data, self.zero_state(batch_data.size(0)))
            loss = self.calc_loss(pred_pi, pred_mu, pred_cov, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if inc_epoch:
                if verbose:
                    print(f"Epoch {epoch} last batch loss: {loss}")
                epoch += 1

    def calc_loss(self, pi, mu, cov, targets) -> torch.Tensor:
        """ Calculates the NLL loss. """
        target_data = targets
        target_data = torch.tile(torch.reshape(target_data, [-1, 1, self.out_seq_len, 2]), [1, self.mixture_components, 1, 1])

        log_pdf_val = torch.reshape(self.__normal_log(target_data, mu, cov, torch.linalg.inv(cov)), [-1, self.mixture_components, self.out_seq_len])  # [batch_size, n_splines, seq_len_2]
        log_pdf_val = torch.sum(log_pdf_val, dim=2)  # [ batch_size, n_splines ]  <- p(x,y,z) = p(x)p(y)p(z) -> ln p(x,y,z) = ln(p(x)) + ln(p(y)) + ln(p(z)) <- vars are independent
        res = torch.add(torch.log(pi), log_pdf_val)
        res = torch.logsumexp(res, dim=1)
        res = torch.multiply(res, -1)

        return torch.mean(res)

    def __normal_log(self, x, m, c, c_inv):
        mahalanobis = torch.squeeze(
            torch.matmul(torch.matmul(torch.reshape(x - m, [-1, self.mixture_components, self.out_seq_len, 1, 2]),
                                      c_inv),
                         torch.reshape(x - m, [-1, self.mixture_components, self.out_seq_len, 2, 1])))
        log_det = torch.log(torch.linalg.det(c) + 1e-6).squeeze()
        return -0.5 * (log_det + mahalanobis + 2 * torch.log(2 * self.__torch_pi))



