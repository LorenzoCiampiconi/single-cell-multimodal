import logging
from typing import Type

import pytorch_lightning as pl
import torch
from torch import nn

from scmm.models.embedding.autoencoder.encoder.base_encoder import FullyConnectedEncoder
from scmm.models.embedding.autoencoder.full.dataset import as_numpy
from scmm.models.embedding.autoencoder.full.trainers import AutoEncoderTrainer
from scmm.models.embedding.autoencoder.full.types.basic import BasicAutoEncoder
from scmm.models.embedding.base_embedder import Embedder

logger = logging.getLogger(__name__)


class BasicAutoEncoderEmbedder(AutoEncoderTrainer, Embedder):
    @property
    def autoencoder_class(self) -> Type[pl.LightningModule]:
        return BasicAutoEncoder

    def build_params(self):
        lr = self.model_params["lr"]

        decoder = None
        if "encoder_sequential" in self.model_params:
            encoder = self.model_params["encoder_sequential"]
            if "decoder_sequential" in self.model_params:
                decoder = self.model_params["decoder_sequential"]
        else:
            shrinking_factors = self.model_params["shrinking_factors"]
            activation_function = self.model_params["activation_function"]

            input_dim = self.input_dim
            latent_dim = self.latent_dim

            dims = [input_dim]
            for shrinking_factor in shrinking_factors:
                hidden_dim = dims[-1] // shrinking_factor
                dims.append(hidden_dim)

            final_dim = dims[-1]

            assert (
                final_dim == latent_dim
            ), f"Latent_dim ({latent_dim}) inconsitent with shinkring factors ({shrinking_factors})"

            encoder_sequential = nn.Sequential(
                *[x for y in [[nn.Linear(i, o), activation_function()] for i, o in zip(dims[:-1], dims[1:])] for x in y]
            )

            encoder = FullyConnectedEncoder(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                fully_connected_sequential=encoder_sequential,
            )

        return dict(lr=lr, encoder=encoder, decoder=decoder, loss=self.model_params["loss"])

    def build_model(self):
        params = self.build_params()
        return self.autoencoder_class(**params)

    def _predict(self, ds):
        out = self.trainer.predict(self.model, ds)
        out = as_numpy(torch.cat(out)).reshape(-1, self.input_dim)
        return out
