# -*- coding: utf-8 -*-
import logging

#%% imports

import torch
import pytorch_lightning as pl

from pytorch_lightning import callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import sparse
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
}


#%% build datasets

train_svd_caching_path = app_static_dir('cache') / 'svd_2000_train.npz'
test_svd_caching_path = app_static_dir('cache') / 'svd_2000_test.npz'

svd = TruncatedSVD(n_components=2000, random_state=config['seed'])

if train_svd_caching_path.is_file():
    logger.info('loading test reduced by svd from file')
    train_ds = sparse.load_npz(train_svd_caching_path)
else:
    train_mat = load_sparse(split="train", problem="cite", type="inputs")
    train_ds = BaseDataset(svd.fit_transform(train_mat))
    sparse.save_npz(train_svd_caching_path, train_ds)

if test_svd_caching_path.is_file():
    logger.info('loading test reduced by svd from file')
    test_ds = sparse.load_npz(train_svd_caching_path)
else:
    test_mat = load_sparse(split="test", problem="cite", type="inputs")
    test_ds = BaseDataset(svd.transform(test_mat))
    sparse.save_npz(test_svd_caching_path, test_ds)


#%% init model

pl.seed_everything(config["seed"], workers=True)

input_dim = 2000
latent_dim = 64
activation_function = nn.SELU
shrinking_factors = (8, 2)
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

logger = TensorBoardLogger(settings["tensorboard"]["log"], name="autoencoder_test")  # , version="0")
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
