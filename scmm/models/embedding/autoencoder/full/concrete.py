import logging

import torch
import pytorch_lightning as pl

from torch import nn
from scmm.models.embedding.autoencoder.full.base import AutoEncoder

from scmm.models.embedding.autoencoder.full.dataset import as_numpy
from scmm.models.embedding.autoencoder.decoder.base import FullyConnectedDecoder
from scmm.models.embedding.autoencoder.encoder.base import FullyConnectedEncoder
from scmm.models.embedding.autoencoder.full.trainer import AutoEncoderTrainer
from scmm.models.embedding.base import Embedder

logger = logging.getLogger(__name__)


class BasicAutoEncoderEmbedder(AutoEncoderTrainer, Embedder):
    def build_model(self):
        pl.seed_everything(self.seed, workers=True)

        input_dim = self.input_dim
        latent_dim = self.latent_dim
        shrinking_factors = self.model_params["shrinking_factors"]
        activation_function = self.model_params["activation_function"]
        lr = self.model_params["lr"]

        hidden_dim = input_dim // shrinking_factors[0]
        final_dim = hidden_dim // shrinking_factors[1]

        assert (
            final_dim == latent_dim
        ), f"Latent_dim ({latent_dim}) inconsitent with shinkring factors ({shrinking_factors})"

        encoder = FullyConnectedEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            fully_connected_sequential=nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_function(),
                nn.Linear(hidden_dim, final_dim),
                activation_function(),
                nn.Linear(final_dim, latent_dim),
                activation_function(),
            ),
        )
        decoder = FullyConnectedDecoder(
            latent_dim=latent_dim,
            fully_connected_sequential=nn.Sequential(
                nn.Linear(latent_dim, final_dim),
                activation_function(),
                nn.Linear(final_dim, hidden_dim),
                activation_function(),
                nn.Linear(hidden_dim, input_dim),
                activation_function(),
            ),
        )

        return AutoEncoder(lr=lr, encoder=encoder, decoder=decoder)

    def _predict(self, ds):
        out = self.trainer.predict(self.model, ds)
        out = as_numpy(torch.cat(out)).reshape(-1, self.input_dim)
        return out
