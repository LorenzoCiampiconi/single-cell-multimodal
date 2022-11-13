import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn

from scmm.problems.metrics import common_metrics
from scmm.utils.metadata import map_to_dataset

original_dim = None

standard_lgbm_cite_conf = {
    "learning_rate": 0.1,
    "objective": "regression",
    "metric": "rmse",  # mae',
    "random_state": 0,
    "reg_alpha": 0.03,
    "reg_lambda": 0.002,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "max_depth": 10,
    "num_leaves": 186,
    "min_child_samples": 263,
}

dataloader_kwargs = {
    "batch_size": 64,
    "num_workers": 4,
}
trainer_kwargs = {
    # "accelerator": "gpu",
    "max_epochs": 1,
    "check_val_every_n_epoch": 1,
    # "val_check_interval": 1,
    "log_every_n_steps": 50,
    "gradient_clip_val": 1,
}

standard_autoencoder_net_params = {
    "lr": 1e-3,
    "shrinking_factors": (4, 4, 2),
    "activation_function": nn.SELU,
    "loss": nn.SmoothL1Loss(),
}

metadata = pd.read_csv("data/metadata.csv")
metadata["dataset"] = metadata.apply(map_to_dataset, axis=1)
train_metadata = metadata[(metadata["technology"] == "citeseq") & (metadata["dataset"] == "train")]

cv_params = {
    "cv": LeaveOneGroupOut(),
    "groups": train_metadata["donor"].to_numpy(),
    "scoring": common_metrics,
    "verbose": 10,
}
