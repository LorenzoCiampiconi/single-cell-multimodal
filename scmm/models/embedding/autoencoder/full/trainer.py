import abc
import logging
from typing import Any, Dict

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


class AutoEncoderTrainer(metaclass=abc.ABCMeta):
    def __init__(
        self, *, seed: int, input_dim: int, output_dim: int, model_params: Dict[str, Any], train_params: Dict[str, Any]
    ):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.latent_dim = output_dim
        self.fitted = False

        self._model_params = model_params
        self._train_params = train_params

        self.model = self.build_model()
        self.init_trainer()

    @abc.abstractmethod
    def build_model(self):
        ...

    @property
    def model_params(self):
        return self._model_params

    @property
    def logger_kwargs(self):
        return self._train_params["logger_kwargs"]

    @property
    def trainer_kwargs(self):
        return self._train_params["trainer_kwargs"]

    @property
    def dataloader_kwargs(self):
        return self._train_params["dataloader_kwargs"]

    def is_fit(self) -> bool:
        return self.fitted

    def init_trainer(self):
        logger = TensorBoardLogger(settings["tensorboard"]["path"], **self.logger_kwargs)
        self.trainer = pl.Trainer(
            deterministic=True,
            logger=logger,
            # accelerator="gpu",
            # max_epochs=10,
            # check_val_every_n_epoch=1,
            # val_check_interval=1,
            # log_every_n_steps=50,
            # gradient_clip_val=1,
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
            **self.trainer_kwargs,
        )

    def build_data_loader(self, mat, shuffle=False):
        ds = BaseDataset(mat)
        dsl = DataLoader(
            ds,
            shuffle=shuffle,
            pin_memory=True,
            **self.dataloader_kwargs
            # batch_size=64,
            # num_workers=0,
        )
        return dsl

    def fit(self, *, input: ArrayLike):
        dsl = self.build_data_loader(input, shuffle=True)
        self.trainer.fit(self.model, train_dataloaders=dsl)
        self.fitted = True
        return self

    def transform(self, *, input: ArrayLike) -> np.array:
        dsl = self.build_data_loader(input)
        encoder = self.model.encoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([encoder(x) for x in dsl])).reshape(-1, self.latent_dim)

        return out
