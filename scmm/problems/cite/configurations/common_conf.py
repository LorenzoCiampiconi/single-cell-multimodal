from torch import nn

from scmm.problems.metrics import common_metrics

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

cv_params = {"cv": 3, "scoring": common_metrics, "verbose": 10}