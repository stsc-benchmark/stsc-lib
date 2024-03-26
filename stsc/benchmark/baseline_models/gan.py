from typing import Union, List
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as tfun

from stsc.benchmark.misc.mixin import BatchGenMixin


class _SeqGenerator2D(torch.nn.Module):
    def __init__(self, embedding_dim, enc_hidden_size, dec_hidden_size, mlp_dim, num_layers, noise_dim, torch_device):
        super(_SeqGenerator2D, self).__init__()

        assert noise_dim > 0, "<noise_dim> needs to be greater 0"

        self.device = torch_device

        self.embedding_dim = embedding_dim
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.noise_dim = noise_dim

        self.enc_embedding = torch.nn.Linear(2, embedding_dim).to(self.device)
        self.encoder = torch.nn.LSTM(input_size=embedding_dim,
                                     hidden_size=enc_hidden_size,
                                     num_layers=num_layers,
                                     bias=True,
                                     batch_first=True).to(self.device)

        mlp_dims = [enc_hidden_size, mlp_dim, dec_hidden_size - noise_dim]
        layers = []
        for dim_in, dim_out in zip(mlp_dims[:-1], mlp_dims[1:]):
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers.append(torch.nn.ReLU())
        self.dec_input_dim_transform = torch.nn.Sequential(*layers).to(self.device)

        self.dec_embedding = torch.nn.Linear(2, embedding_dim).to(self.device)
        self.decoder = torch.nn.LSTM(input_size=embedding_dim,
                                     hidden_size=dec_hidden_size,
                                     num_layers=num_layers,
                                     bias=True,
                                     batch_first=True).to(self.device)
        self.readout = torch.nn.Linear(dec_hidden_size, 2).to(self.device)

    def zero_state(self, batch_size: int, hidden_size: int):
        # Note: although batch_first is set to True, the hidden state is still expected to be [ layers, bs, dim ]
        return (torch.zeros([self.num_layers, batch_size, hidden_size]).to(self.device),  # h
                torch.zeros([self.num_layers, batch_size, hidden_size]).to(self.device))  # c

    def forward(self, observation, num_steps, scale_factor=None):
        # x shape is [ batch_size, seq_len, 2 ]
        batch_size = observation.size(0)

        # convert input to offset sequences
        obs_zero = torch.cat([torch.zeros([batch_size, 1, 2]).to(observation), observation], dim=1)
        obs_offsets = obs_zero[:, 1:] - obs_zero[:, :-1]

        # Encode seq
        enc_in_embed = self.enc_embedding(obs_offsets.reshape(-1, 2))
        _, final_state = self.encoder(enc_in_embed.reshape(batch_size, -1, self.embedding_dim), self.zero_state(batch_size, self.enc_hidden_size))
        final_h = final_state[0]

        # transform encoder output dimension
        dec_input_transform_input = final_h.reshape(-1, self.enc_hidden_size)
        noise_input = self.dec_input_dim_transform(dec_input_transform_input)

        # Add Noise
        z = torch.randn([noise_input.size(0), self.noise_dim], device=self.device)
        decoder_h = torch.cat([noise_input, z], dim=1).unsqueeze(0)

        decoder_c = self.zero_state(batch_size, self.dec_hidden_size)[1]

        # Predict Trajectory
        dec_state = (decoder_h, decoder_c)
        dec_input = obs_offsets[:, -1:]  # use last offset as input

        # Predict Trajectory
        pred_off = torch.zeros([batch_size, num_steps, 2], dtype=torch.float32, device=self.device)
        for i in range(num_steps):
            inp_embed = self.dec_embedding(dec_input.reshape(-1, 2))
            out, dec_state = self.decoder(inp_embed.reshape(batch_size, 1, self.embedding_dim), dec_state)
            next_off = self.readout(torch.reshape(out, [-1, self.dec_hidden_size]))
            pred_off[:, i, :] = next_off
            dec_input = next_off.reshape(batch_size, 1, 2)

        full_traj_off = torch.cat([obs_offsets, pred_off], dim=1)
        full_traj = torch.cumsum(full_traj_off, dim=1)

        return full_traj[:, observation.size(1):]


class _SeqDiscriminator2D(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, mlp_dim, torch_device):
        super(_SeqDiscriminator2D, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch_device

        self.enc_embedding = torch.nn.Linear(2, embedding_dim).to(self.device)
        self.encoder = torch.nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bias=True,
                                     batch_first=True).to(self.device)

        classifier_dims = [hidden_size, mlp_dim, 1]
        layers = []
        for dim_in, dim_out in zip(classifier_dims[:-1], classifier_dims[1:]):
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers.append(torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(*layers).to(self.device)  # classifies input as real or fake

    def zero_state(self, batch_size: int, hidden_size: int):
        # Note: although batch_first is set to True, the hidden state is still expected to be [ layers, bs, dim ]
        return (torch.zeros([self.num_layers, batch_size, hidden_size]).to(self.device),  # h
                torch.zeros([self.num_layers, batch_size, hidden_size]).to(self.device))  # c

    def forward(self, trajectories):  # TODO: switch to offsets?
        # x shape is [ batch_size, seq_len, 2 ]
        batch_size = trajectories.shape[0]

        # convert input to offset sequences
        trajs_zero = torch.cat([torch.zeros([batch_size, 1, 2]).to(trajectories), trajectories], dim=1)
        trajs_offsets = trajs_zero[:, 1:] - trajs_zero[:, :-1]

        enc_in_embed = self.enc_embedding(trajs_offsets.reshape(-1, 2))
        _, final_state = self.encoder(enc_in_embed.reshape(batch_size, -1, self.embedding_dim), self.zero_state(batch_size, self.hidden_size))

        final_h = final_state[0]
        return self.classifier(final_h.squeeze())


class GANSequencePredictor2D(torch.nn.Module, BatchGenMixin):
    def __init__(self, embedding_dim=16, gen_enc_hidden_size=32, gen_dec_hidden_size=48, noise_dim=8, discr_hidden_size=32, num_layers=1, mlp_dim=64, torch_device=None):
        super(GANSequencePredictor2D, self).__init__()

        if torch_device is None:
            torch_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        self.device = torch_device

        self.generator = _SeqGenerator2D(embedding_dim, gen_enc_hidden_size, gen_dec_hidden_size, mlp_dim, num_layers, noise_dim, torch_device)
        self.discriminator = _SeqDiscriminator2D(embedding_dim, discr_hidden_size, num_layers, mlp_dim, torch_device)

    def save(self, base_path):
        torch.save(self.state_dict(), f"{base_path}.pth")
        with open(f"{base_path}.meta", "wb") as f:
            pkl.dump({
                "embedding_dim": self.generator.embedding_dim,
                "gen_enc_hidden_size": self.generator.enc_hidden_size,
                "gen_dec_hidden_size": self.generator.dec_hidden_size,
                "noise_dim": self.generator.noise_dim,
                "discr_hidden_size": self.discriminator.hidden_size,
                "num_layers": self.generator.num_layers,
                "mlp_dim": self.generator.mlp_dim,
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

    # predict function with num_samples
    def predict(self, observation, num_steps, num_samples=20, scale_factor=None):
        if type(observation) == np.ndarray:
            observation = torch.tensor(observation).float().to(self.device)

        if len(observation.shape) == 2:
            observation = observation.unsqueeze(0)

        assert len(observation.shape) == 3, "observation shape needs to be 3-dimensional: [ batch_size, obs_len, 2 ]."

        obs_rep = torch.repeat_interleave(observation, num_samples, dim=0)
        pred = self.generator(obs_rep, num_steps)

        prediction = torch.reshape(pred, [observation.shape[0], num_samples, num_steps, 2]).data.cpu().numpy()
        if scale_factor is not None:
            prediction = prediction * scale_factor
        return prediction

    def estimate_parameters(
            self, 
            data: Union[List[np.ndarray],np.ndarray], 
            batch_size: int,
            obs_len: int, 
            pred_len: int, 
            num_samples: int = 10, 
            lr_generator: float = 1e-4, 
            lr_discriminator: float = 1e-3, 
            n_epochs: int = 1000,
            verbose: bool = False
        ) -> None:
        d_steps = 2
        g_steps = 1

        ## use entire dataset at once
        #if type(dataset) == Dataset:
        #    batch_data = dataset.all_slices(obs_len + pred_len, torch_tensor=True, torch_device=self.device)
        #else:
        #    batch_data = torch.tensor(dataset).float().to(self.device)
        batch_gen = self.build_batch_generator(data, batch_size, obs_len + pred_len)

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_generator)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)

        loss_d = 9999.
        loss_g = 9999.

        d_steps_left = d_steps
        g_steps_left = g_steps
        epoch = 0
        while epoch < n_epochs:
            batch_data, inc_epoch = next(batch_gen)
            if d_steps_left > 0:
                loss_d = self.__discriminator_step(batch_data, obs_len, pred_len, optimizer_d)
                d_steps_left -= 1

            elif g_steps_left > 0:
                loss_g, ade = self.__generator_step(batch_data, obs_len, pred_len, optimizer_g, num_samples)
                g_steps_left -= 1

            if d_steps_left == 0 and g_steps_left == 0:
                d_steps_left = d_steps
                g_steps_left = g_steps
                if verbose:
                    print(f"[Epoch {epoch}] - d_loss: {loss_d} | g_loss: {loss_g} | ade: {ade}")

            if inc_epoch:
                epoch += 1

    def __discriminator_step(self, batch_data, obs_len, pred_len, optimizer):
        optimizer.zero_grad()

        pred = self.generator(batch_data[:, :obs_len], pred_len)  # , train_in=batch_data[:, obs_len-1:-1])
        obs_pred = torch.cat([batch_data[:, :obs_len], pred], dim=1)

        # Measure discriminator's ability to classify real from generated samples
        scores_fake = self.discriminator(obs_pred)
        y_fake = torch.zeros_like(scores_fake)
        scores_real = self.discriminator(batch_data)
        y_real = torch.ones_like(scores_real)
        loss_fake = tfun.binary_cross_entropy_with_logits(scores_fake, y_fake)
        loss_real = tfun.binary_cross_entropy_with_logits(scores_real, y_real)

        d_loss = loss_real + loss_fake

        d_loss.backward()
        optimizer.step()

        return d_loss.detach()

    def __generator_step(self, batch_data, obs_len, pred_len, optimizer, num_samples, clip_thresh=2.):
        batch_size = batch_data.size(0)

        l2_losses = torch.zeros([batch_data.size(0), num_samples], device=self.device)
        all_obs_pred = torch.zeros([batch_data.size(0), num_samples, obs_len+pred_len, 2], device=self.device)
        for i in range(num_samples):
            pred = self.generator(batch_data[:, :obs_len], pred_len)  # , train_in=batch_data[:, obs_len-1:-1])
            obs_pred = torch.cat([batch_data[:, :obs_len], pred], dim=1)

            l2_losses[:, i] = torch.mean(torch.sqrt(torch.sum(torch.square(pred - batch_data[:, obs_len:]), dim=2)),dim=1)
            all_obs_pred[:, i] = obs_pred

        variety_loss, min_inds = torch.min(l2_losses, dim=1)
        variety_loss = variety_loss.mean()

        min_err_preds = torch.gather(all_obs_pred, dim=1, index=min_inds.reshape(-1, 1, 1, 1).tile(1, 1, all_obs_pred.size(2), all_obs_pred.size(3)))
        scores_fake = self.discriminator(min_err_preds.reshape(batch_size, -1, 2))
        y_fake = torch.zeros_like(scores_fake)
        adversarial_loss = tfun.binary_cross_entropy_with_logits(scores_fake, y_fake)

        loss = adversarial_loss + variety_loss

        ade = torch.mean(l2_losses)

        optimizer.zero_grad()
        loss.backward()
        if clip_thresh > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), clip_thresh
            )
        optimizer.step()

        return loss.detach(), ade.detach()
