from scmm.models.embedding.autoencoder import BasicAutoEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.metrics import common_metrics
from torch import nn

model_label = "lgbm_w_autoencoder_ago"
model_class = LGBMwMultilevelEmbedderCite
seed = 0
original_dim = None

model_params = {
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

cv_params = {"cv": 3, "scoring": common_metrics, "verbose": 10}

logger_kwargs = {
    "name": "basic_autoencoder",
}
dataloader_kwargs = {
    "batch_size": 64,
    "num_workers": 4,
}
trainer_kwargs = {
    "accelerator": "gpu",
    "max_epochs": 20,
    "check_val_every_n_epoch": 1,
    # "val_check_interval": 1,
    "log_every_n_steps": 50,
    "gradient_clip_val": 1,
}
net_params = {
    "lr": 1e-3,
    "shrinking_factors": (2, 2),
    "activation_function": nn.ReLU,
}
embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": 200,
    "embedders_config": [
        (
            TruncatedSVDEmbedder,
            {
                "seed": seed,
                "input_dim": original_dim,
                "output_dim": 800,
            },
        ),
        (
            BasicAutoEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": 800,
                "output_dim": 200,
                "model_params": net_params,
                "train_params": {
                    "logger_kwargs": logger_kwargs,
                    "dataloader_kwargs": dataloader_kwargs,
                    "trainer_kwargs": trainer_kwargs,
                },
            },
        ),
    ],
}

configuration = {
    "cv_params": cv_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "seed": seed,
}
