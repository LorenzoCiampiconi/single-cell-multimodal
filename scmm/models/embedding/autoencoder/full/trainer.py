import logging

import numpy as np
import torch
import pytorch_lightning as pl

from numpy.typing import ArrayLike
from pytorch_lightning import callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from scmm.models.embedding.autoencoder.full.dataset import BaseDataset, as_numpy
from scmm.utils.log import settings

logger = logging.getLogger(__name__)


class AutoEncoderTrainer:
    fitted: bool
    max_epochs: int

    def is_fit(self) -> bool:
        return self.fitted

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

    def _build_data_loader(self, mat, shuffle=False):
        ds = BaseDataset(mat)
        dsl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        return dsl

    def transform(self, *, input: ArrayLike) -> np.array:
        dsl = self._build_data_loader(input)
        encoder = self.model.encoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([encoder(x) for x in dsl])).reshape(-1, self.latent_dim)

        return out
