# -*- coding: utf-8 -*-

from typing import Optional
from torch import nn

from scmm.models.embedding.autoencoder.utils import init_fc_snn


class ExtraFeatureDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, output_dim: int, model: Optional[nn.Module] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        if model is None:
            self.model = nn.Linear(self.latent_dim, self.output_dim)
        else:
            self.model = model

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_fc_snn)

    def forward(self, x):
        return self.model(x)


class JointDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, output_dim: int, decoder: nn.Module, head: Optional[nn.Module] = None):
        super().__init__()
        self.decoder = decoder
        self.features = ExtraFeatureDecoder(latent_dim, output_dim, model=head)

    def forward(self, x):
        return self.decoder(x), self.features(x)
