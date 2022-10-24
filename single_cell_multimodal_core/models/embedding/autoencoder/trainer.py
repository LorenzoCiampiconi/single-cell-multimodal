import logging

import numpy as np
import torch
import pytorch_lightning as pl

from numpy.typing import ArrayLike
from pytorch_lightning import callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from single_cell_multimodal_core.models.embedding.autoencoder.base import AutoEncoder

from single_cell_multimodal_core.models.embedding.dataset import BaseDataset, as_numpy
from single_cell_multimodal_core.models.embedding.decoder.base import FullyConnectedDecoder
from single_cell_multimodal_core.models.embedding.encoder.base import FullyConnectedEncoder
from single_cell_multimodal_core.utils.log import settings

logger = logging.getLogger(__name__)


class AutoEncoderTrainer:
    seed: int

    def __init__(self, input_dim, latent_dim, seed):
        # structural
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.shrinking_factors = (8, 2)
        self.activation_function = nn.SELU

        # training
        self.lr = 0.001
        self.batch_size = 64
        self.num_workers = 0
        self.max_epochs = 10
        self.name = "autoencoder_test"

        self.seed = seed  # todo remove

        self._fit = False

        self.model = None

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

        self.model = AutoEncoder(lr=self.lr, encoder=encoder, decoder=decoder)

    def _build_data_loader(self, mat):
        self.ds = BaseDataset(mat)
        self.dsl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def _init_trainer(self):
        logger = TensorBoardLogger(settings["tensorboard"]["path"], name=self.name)
        self.trainer = pl.Trainer(
            # accelerator="gpu",
            # devices=-1,
            max_epochs=self.max_epochs,
            deterministic=True,
            logger=logger,
            check_val_every_n_epoch=1,
            val_check_interval=0.5,
            log_every_n_steps=25,
            gradient_clip_val=1,
            # detect_anomaly=True,
            # track_grad_norm=2,
            # limit_train_batches=0.5,
            callbacks=[
                cbs.ModelCheckpoint(save_top_k=-1),
                # cbs.LearningRateMonitor("step"),
                # cbs.StochasticWeightAveraging(
                #     swa_epoch_start=0.75,
                #     swa_lrs=5e-3,
                #     # annealing_epochs=3,
                # ),
                # cbs.GPUStatsMonitor(),
                # cbs.EarlyStopping(),
                # cbs.RichProgressBar(),
                # cbs.RichModelSummary(),
            ],
        )

    def is_fit(self) -> bool:
        return self._fit

    def fit(self, mat: ArrayLike):
        self._build_model()
        self._init_trainer()
        self._build_data_loader(mat)
        self.trainer.fit(self.model, train_dataloaders=self.dsl)
        self._fit = True

    def _predict(self, ds):
        out = self.trainer.predict(self.model, ds)
        out = as_numpy(torch.cat(out)).reshape(-1, self.input_dim)
        return out

    def transform(self, mat) -> np.array:
        ds = BaseDataset(mat)
        dslt = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        encoder = self.model.encoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([encoder(x) for x in dslt])).reshape(-1, self.latent_dim)

        return out
