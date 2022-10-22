# -*- coding: utf-8 -*-

#%% imports

import logging

import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning import callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import TruncatedSVD
from torch import nn
from torch.utils.data import DataLoader
from single_cell_multimodal_core.data_handling import load_sparse
from single_cell_multimodal_core.models.embedding.autoencoder.base import AutoEncoder

from single_cell_multimodal_core.models.embedding.dataset import BaseDataset, as_numpy
from single_cell_multimodal_core.models.embedding.encoder.base import FullyConnectedEncoder
from single_cell_multimodal_core.utils.appdirs import app_static_dir
from single_cell_multimodal_core.utils.log import settings, setup_logging

logger = logging.getLogger(__name__)

setup_logging()

#%% setup

model_name = "autoencoder_test"
config = {
    "lr": 1e-3,
    "batch_size": 64,
    "num_workers": 0,
    "seed": 42,
    "max_epochs": 10,
    "n_components": 64,
    "latent_dim": 16,
    "shrinking_factors": (8, 2),
}


#%% build datasets

n_components = config["n_components"]

svd_caching_path = app_static_dir("cache") / f"svd_{n_components}.npz"

if svd_caching_path.exists():
    logger.info("loading inputs reduced by svd from file")
    with np.load(svd_caching_path) as npz_file:
        train_mat, test_mat = npz_file["train_mat"], npz_file["test_mat"]
else:
    logger.info(f"compute svd with {n_components} components")
    svd = TruncatedSVD(n_components=n_components, random_state=config["seed"])
    train_mat = load_sparse(split="train", problem="cite", type="inputs")
    test_mat = load_sparse(split="test", problem="cite", type="inputs")
    train_mat = svd.fit_transform(train_mat)
    test_mat = svd.transform(test_mat)
    np.savez_compressed(svd_caching_path, train_mat=train_mat, test_mat=test_mat)


train_ds = BaseDataset(train_mat)
test_ds = BaseDataset(test_mat)


#%% init model

pl.seed_everything(config["seed"], workers=True)

input_dim = n_components
latent_dim = config["latent_dim"]
activation_function = nn.SELU
shrinking_factors = config["shrinking_factors"]
hidden_dim = input_dim // shrinking_factors[0]

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
decoder = None

model = AutoEncoder(lr=config["lr"], encoder=encoder, decoder=decoder)

#%% init trainer

logger = TensorBoardLogger(settings["tensorboard"]["path"], name="autoencoder_test")  # , version="0")
dsl = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    pin_memory=True,
    num_workers=config["num_workers"],
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    max_epochs=config["max_epochs"],
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

# lr_finder = trainer.tuner.lr_find(model, tdsl)
# lr_finder.plot(suggest=True).show()
# new_lr = lr_finder.suggestion()

#%% fit

trainer.fit(model, train_dataloaders=dsl)

#%% predict

out = trainer.predict(model, test_ds)
# out = as_numpy(torch.cat(out)) for convinience

#%% use single layer example (extract embedding)

encoder = model.encoder.eval()

with torch.no_grad():
    out = as_numpy(torch.cat([encoder(x) for x in test_ds]))

#%% manual test

index = 2100

x = test_ds[index]

with torch.no_grad():
    inputs = x.unsqueeze(0)
    z = model.encoder(x)
    out = model(x)

# use as_numpy to inspect the outputs
