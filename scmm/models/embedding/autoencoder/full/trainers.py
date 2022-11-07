import abc
import logging
from typing import Any, Dict, Type

import numpy as np
import torch
import pytorch_lightning as pl

from numpy.typing import ArrayLike
from pytorch_lightning import callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from scmm.models.embedding.autoencoder.full.dataset import BaseDataset, IODataset, as_numpy
from scmm.models.embedding.base_embedder import Embedder
from scmm.utils.log import settings

logger = logging.getLogger(__name__)


class AutoEncoderTrainer(Embedder, metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        seed: int,
        input_dim: int,
        output_dim: int,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        save_checkpoints=True,
    ):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.latent_dim = output_dim

        self._model_params = model_params
        self._train_params = train_params

        pl.seed_everything(self.seed, workers=True)

        self._save_checkpoints = save_checkpoints

        self.model = self.build_model()
        self.trainer = None

    @abc.abstractmethod
    def build_model(self):
        ...

    @property
    @abc.abstractmethod
    def autoencoder_class(self) -> Type[pl.LightningModule]:
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

    def init_trainer(self, save_checkpoints=True, **kwargs):
        callbacks = []
        if save_checkpoints:
            callbacks.append(cbs.ModelCheckpoint(save_top_k=-1))

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
            callbacks=callbacks,
            **self.trainer_kwargs,
        )

    def build_data_loader(self, input, *, shuffle=False, **kwargs):
        ds = BaseDataset(input)
        dsl = DataLoader(
            ds,
            shuffle=shuffle,
            pin_memory=True,
            **self.dataloader_kwargs
            # batch_size=64,
            # num_workers=0,
        )
        return dsl

    def fit(self, *, input: ArrayLike, **kwargs):
        self.init_trainer(**kwargs)
        dsl = self.build_data_loader(input, shuffle=True)
        self.trainer.fit(self.model, train_dataloaders=dsl)
        self._fitted = True
        return self

    def transform(self, *, input: ArrayLike, **kwargs) -> np.array:
        dsl = self.build_data_loader(input)
        encoder = self.model.encoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([encoder(x) for x in dsl])).reshape(-1, self.latent_dim)
        logger.info("Autoencoder has transformed the input")
        return out

    def inverse_transform(self, *, input: ArrayLike, **kwargs) -> np.array:
        dsl = self.build_data_loader(input)
        decoder = self.model.decoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([decoder(x) for x in dsl])).reshape(-1, self.input_dim)
        logger.info("Autoencoder has inverse transformed the input")
        return out

    def save_model(self, path, **kwargs):
        self.trainer.save_checkpoint(path, **kwargs)

    def load_model(self, path, **kwargs):
        return self.autoencoder_class.load_from_checkpoint(path, **kwargs)


class MultiTaskAutoEncoderTrainer(AutoEncoderTrainer, metaclass=abc.ABCMeta):
    def build_data_loader(self, input, *, Y=None, shuffle=False, **kwargs):
        assert Y is not None
        ds = IODataset(input, Y)
        dsl = DataLoader(
            ds,
            shuffle=shuffle,
            pin_memory=True,
            **self.dataloader_kwargs
            # batch_size=64,
            # num_workers=0,
        )
        return dsl

    def fit(self, *, input: ArrayLike, Y=None, **kwargs):
        assert Y is not None
        self.init_trainer(**kwargs)
        dsl = self.build_data_loader(input, Y=Y, shuffle=True)
        self.trainer.fit(self.model, train_dataloaders=dsl)
        self._fitted = True
        return self

    def transform(self, *, input: ArrayLike, **kwargs) -> np.array:
        dsl = super().build_data_loader(input)
        encoder = self.model.encoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([encoder(x) for x in dsl])).reshape(-1, self.latent_dim)
        logger.info("Autoencoder has transformed the input")
        return out

    def inverse_transform(self, *, input: ArrayLike, **kwargs) -> np.array:
        dsl = super().build_data_loader(input)
        decoder = self.model.decoder.decoder.eval()

        with torch.no_grad():
            out = as_numpy(torch.cat([decoder(x) for x in dsl])).reshape(-1, self.input_dim)
        logger.info("Autoencoder has inverse transformed the input")
        return out
