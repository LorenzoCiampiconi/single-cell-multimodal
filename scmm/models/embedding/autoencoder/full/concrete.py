import logging
from typing import Any, Dict

import torch
import pytorch_lightning as pl

from numpy.typing import ArrayLike
from torch import nn
from scmm.models.embedding.autoencoder.full.base import AutoEncoder

from scmm.models.embedding.autoencoder.full.dataset import as_numpy
from scmm.models.embedding.autoencoder.decoder.base import FullyConnectedDecoder
from scmm.models.embedding.autoencoder.encoder.base import FullyConnectedEncoder
from scmm.models.embedding.autoencoder.full.trainer import AutoEncoderTrainer
from scmm.models.embedding.base import Embedder

logger = logging.getLogger(__name__)


class AutoEncoderEmbedder(AutoEncoderTrainer, Embedder):
    def __init__(
        self, *, seed: int, input_dim: int, output_dim: int, model_kwargs: Dict[str, Any], train_kwargs: Dict[str, Any]
    ):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.latent_dim = output_dim
        self.fitted = False

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs

        # model
        self.shrinking_factors = (8, 2)
        self.activation_function = nn.SELU

        # training
        self.lr = 0.001
        self.batch_size = 64
        self.num_workers = 0
        self.max_epochs = 10
        self.name = "basic_autoencoder_test"

        self.model = self._build_model()
        self._init_trainer()

    def _build_model(self):
        pl.seed_everything(self.seed, workers=True)

        input_dim = self.input_dim
        latent_dim = self.latent_dim
        shrinking_factors = self.shrinking_factors
        hidden_dim = input_dim // shrinking_factors[0]
        activation_function = self.activation_function

        encoder = FullyConnectedEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            fully_connected_sequential=nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_function(),
                nn.Linear(hidden_dim, hidden_dim // (shrinking_factors[1])),
                activation_function(),
                nn.Linear(hidden_dim // shrinking_factors[1], latent_dim),
                activation_function(),
            ),
        )
        decoder = FullyConnectedDecoder(
            latent_dim=latent_dim,
            fully_connected_sequential=nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // shrinking_factors[1]),
                activation_function(),
                nn.Linear(hidden_dim // (shrinking_factors[1]), hidden_dim),
                activation_function(),
                nn.Linear(hidden_dim, input_dim),
                activation_function(),
            ),
        )

        return AutoEncoder(lr=self.lr, encoder=encoder, decoder=decoder)

    def _predict(self, ds):
        out = self.trainer.predict(self.model, ds)
        out = as_numpy(torch.cat(out)).reshape(-1, self.input_dim)
        return out

    def fit(self, *, input: ArrayLike):
        dsl = self._build_data_loader(input, shuffle=True)
        self.trainer.fit(self.model, train_dataloaders=dsl)
        self.fitted = True
