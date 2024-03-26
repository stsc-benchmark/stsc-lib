from typing import Union, List
import pickle as pkl

import numpy as np
import torch

from stsc.benchmark.misc.mixin import BatchGenMixin


class VAESeqPredictor2D(torch.nn.Module, BatchGenMixin):
    def __init__(
            self, 
            train_embed_dim: int = 64,
            train_enc_hidden_dim: int = 128,
            train_enc_num_layers: int = 1,
            input_embed_dim: int = 32,
            encoder_hidden_dim: int = 48,
            encoder_num_layers: int = 1,
            decoder_hidden_dim: int = 48,
            decoder_num_layers: int = 1,
            latent_dim: int = 64, 
            offset_mode: bool = False,
            torch_device="cuda:0"
        ) -> None:
        super(VAESeqPredictor2D, self).__init__()

        self.train_embed_dim = train_embed_dim
        self.train_enc_hidden_dim = train_enc_hidden_dim
        self.train_enc_num_layers = train_enc_num_layers
        self.input_embed_dim = input_embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_layers = decoder_num_layers
        self.latent_dim = latent_dim
        self.offset_mode = offset_mode
        self.device = torch_device

        self.train_input_embed = torch.nn.Linear(2, train_embed_dim, bias=True).to(self.device)
        self.train_input_embed_activation = torch.nn.ReLU().to(self.device)
        self.train_enc_lstm = torch.nn.LSTM(input_size=train_embed_dim,
                                            hidden_size=train_enc_hidden_dim,
                                            num_layers=train_enc_num_layers,
                                            bias=True,
                                            batch_first=True).to(self.device)
        self.train_enc_latent = torch.nn.Linear(train_enc_hidden_dim, 2 * self.latent_dim).to(self.device)

        self.input_embed = torch.nn.Linear(2, input_embed_dim).to(self.device)
        self.encoder_lstm = torch.nn.LSTM(input_size=input_embed_dim,
                                          hidden_size=encoder_hidden_dim,
                                          num_layers=encoder_num_layers,
                                          bias=True,
                                          batch_first=True).to(self.device)
        self.input_embed_activation = torch.nn.ReLU().to(self.device)
        self.concat_transform = torch.nn.Linear(encoder_hidden_dim + self.latent_dim, self.latent_dim).to(self.device)
        self.concat_transform_activation = torch.nn.ReLU().to(self.device)
        self.decoder_lstm = torch.nn.LSTM(input_size=self.latent_dim,
                                          hidden_size=decoder_hidden_dim,
                                          num_layers=decoder_num_layers,
                                          bias=True,
                                          batch_first=True).to(self.device)
        self.decoder_readout = torch.nn.Linear(decoder_hidden_dim, 2, bias=True).to(self.device)

    def save(self, base_path):
        torch.save(self.state_dict(), f"{base_path}.pth")
        with open(f"{base_path}.meta", "wb") as f:
            pkl.dump({
                "train_embed_dim": self.train_embed_dim,
                "train_enc_hidden_dim": self.train_enc_hidden_dim,
                "train_enc_num_layers": self.train_enc_num_layers,
                "input_embed_dim": self.input_embed_dim,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "decoder_num_layers": self.decoder_num_layers,
                "latent_dim": self.latent_dim, 
                "offset_mode": self.offset_mode,
                "torch_device": self.device
            }, f)

    @classmethod
    def from_file(cls, base_path, map_location=None):
        with open(f"{base_path}.meta", "rb") as f:
            kwargs_dict = pkl.load(f)
        if map_location is not None:
            kwargs_dict["torch_device"] = map_location
        instance = cls(**kwargs_dict)
        
        if map_location is None:
            map_location = instance.device        
        instance.load_state_dict(torch.load(f"{base_path}.pth", map_location=map_location))

        return instance

    def forward(self, observation, latent_input, num_steps, is_training=False):
        batch_size, obs_len, _ = observation.shape

        if is_training:
            # get conditional mean and variance of latent variable
            train_embed = self.train_input_embed_activation(self.train_input_embed(latent_input.reshape(-1, 2)))  # full seq?
            train_out, _ = self.train_enc_lstm(train_embed.reshape(batch_size, obs_len + num_steps, -1))
            z_mean_var = self.train_enc_latent(train_out[:, -1])

            # draw sample for z
            z_mean = z_mean_var[:, :self.latent_dim]
            z_log_var = z_mean_var[:, self.latent_dim:]
            epsilon = torch.randn(size=[batch_size, self.latent_dim], device=self.device)
            z = z_mean + torch.exp(z_log_var / 2) * epsilon
        else:
            z = torch.randn(size=[batch_size, self.latent_dim], device=self.device)

        # observation encoder
        embed = self.input_embed_activation(self.input_embed(observation.reshape(-1, 2)))
        enc_out, _ = self.encoder_lstm(embed.reshape(batch_size, obs_len, -1))

        # decoder (prediction generation)
        dec_trans_in = torch.cat([enc_out[:, -1], z], dim=1)
        dec_trans = self.concat_transform_activation(self.concat_transform(dec_trans_in))
        dec_lstm_in = torch.tile(dec_trans.unsqueeze(1), [1, num_steps, 1])
        dec_out, _ = self.decoder_lstm(dec_lstm_in)
        flat_readout = self.decoder_readout(dec_out.reshape(batch_size * num_steps, -1))
        pred = flat_readout.reshape(batch_size, num_steps, 2)

        if is_training:
            return pred, z_mean_var
        return pred

    def estimate_parameters(
            self, 
            data: Union[List[np.ndarray], np.ndarray], 
            batch_size: int, 
            obs_len: int, 
            pred_len: int, 
            num_samples: int = 10, 
            learning_rate: float = 1e-4, 
            n_epochs: int = 1000, 
            verbose: bool = False
        ) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        batch_gen = self.build_batch_generator(data, batch_size, obs_len + pred_len)            

        epoch = 0
        while epoch < n_epochs:
            batch_data, inc_epoch = next(batch_gen)
            batch_size = batch_data.size(0)

            if self.offset_mode:
                ref_pt = batch_data[:, :1]
                ref_pt_pred = batch_data[:, obs_len-1:obs_len]
                batch_data = batch_data[:, 1:] - batch_data[:, :-1]
                obs_len -= 1

            # repeat data such that multiple prediction are generated for each (duplicated) input
            batch_data_rep = torch.repeat_interleave(batch_data, num_samples, dim=0)  # [ batch_size * num_samples, obs_len+pred_len, 2 ]
            pred, z_mean_var = self.forward(batch_data_rep[:, :obs_len], latent_input=batch_data_rep, num_steps=pred_len, is_training=True)

            if self.offset_mode:
                ref_pt_pred_rep = torch.repeat_interleave(ref_pt_pred, num_samples, dim=0)
                ref_pt_rep = torch.repeat_interleave(ref_pt, num_samples, dim=0)
                pred = torch.cat([ref_pt_pred_rep, pred], dim=1).cumsum(dim=1)[:, 1:]
                batch_data_rep = torch.cat([ref_pt_rep, batch_data_rep], dim=1).cumsum(dim=1)
                obs_len += 1

            # reshape pred such that num_samples have own dimension: [ batch_size, num_samples, pred_len, 2 ]
            # -> loss will use the samples for each input
            pred = torch.reshape(pred, [batch_size, num_samples, pred_len, 2])

            # calculate bms loss
            rdiff = torch.mean(torch.square(pred - batch_data_rep[:, obs_len:].reshape(pred.shape)), dim=(2, 3))
            rdiff_min, _ = torch.min(rdiff, dim=1)
            bms_loss = torch.mean(rdiff_min)

            # calculate KL activity regularization
            z_mean = z_mean_var[:, :self.latent_dim]
            z_log_var = z_mean_var[:, self.latent_dim:]
            kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), dim=-1)
            kl_loss = torch.mean(kl_loss)

            # ELBO loss (variation)
            loss = bms_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if inc_epoch:
                epoch += 1
                if verbose:
                    print(f"Epoch {epoch} loss: {loss}")

    def predict(self, observation, num_steps, num_samples=100, scale_factor=None):
        if type(observation) == np.ndarray:
            observation = torch.tensor(observation).float().to(self.device)

        if len(observation.shape) == 2:
            observation = observation.unsqueeze(0)

        assert len(observation.shape) == 3, "observation shape needs to be 3-dimensional: [ batch_size, obs_len, 2 ]."

        # TODO: offset mode
        if self.offset_mode:
            ref_pt_rep = torch.repeat_interleave(observation[:, -1:], num_samples, dim=0)
            observation = observation[:, 1:] - observation[:, :-1]

        obs_rep = torch.repeat_interleave(observation, num_samples, dim=0)
        dummy_latent_input = torch.zeros([obs_rep.shape[0], observation.shape[1] + num_samples, 2], device=self.device)
        pred = self.forward(obs_rep, dummy_latent_input, num_steps, is_training=False)

        if self.offset_mode:
            pred = torch.cat([ref_pt_rep, pred], dim=1).cumsum(dim=1)[:, 1:]

        prediction = torch.reshape(pred, [observation.shape[0], num_samples, num_steps, 2]).data.cpu().numpy()
        if scale_factor is not None:
            prediction = prediction * scale_factor
        return prediction
