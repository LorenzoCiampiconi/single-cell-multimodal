import abc

import torch
from torch import nn, distributions as td

from scmm.models.embedding.autoencoder.encoder.base_encoder import FullyConnectedEncoder


class GaussianFullyConnectedEncoder(FullyConnectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        (
            self._encoding_layer,
            self._encoding_head_mu,
            self._encoding_head_log_var,
        ) = self._instantiate_gaussian_multi_head()

        self.reset_parameters()

    def _instantiate_gaussian_multi_head(self):
        return (
            self.fc_model[:-2],
            self.fc_model[-2:],
            nn.Sequential(
                nn.Linear(20, self.latent_dim),
                nn.SELU(),
            ),
        )  # todo check

    def _forward_backbone(self, x):
        return self._encoding_layer(x)

    def forward(self, x):
        h = self._forward_backbone(x)
        mu = self._encoding_head_mu(h)
        return mu

    def forward_dist(self, x):
        h = self._forward_backbone(x)
        return self.forward_sample(h)

    def forward_sample(self, h):
        mu = self._encoding_head_mu(h)
        logvar = self._encoding_head_log_var(h)

        q = td.Normal(mu, torch.exp(4 * logvar))
        z = q.rsample()

        return q, z
