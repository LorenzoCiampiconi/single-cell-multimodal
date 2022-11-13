from torch import nn

from scmm.models.embedding.autoencoder.full.concrete.multitask import MultiTaskEncoderEmbedder
from scmm.problems.cite.configurations.common_conf import (
    cv_params,
    standard_lgbm_cite_conf,
    original_dim,
    dataloader_kwargs,
)
from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.lgbm_w_autoencoder import logger_kwargs

model_label = "lgbm_w_svd_baseline"
model_class = LGBMwMultilevelEmbedderCite
seed = 0

latent_dim = 128

trainer_kwargs = {
    # "accelerator": "gpu",
    "max_epochs": 10,
    "check_val_every_n_epoch": 1,
    # "val_check_interval": 1,
    "log_every_n_steps": 50,
    "gradient_clip_val": 1,
}

net_params = {
    "lr": 5e-4,
    "shrinking_factors": (2, 2, 2, 2),
    "activation_function": nn.SELU,
    "input_coef": 1 / 10,
    "features_dim": 140,
    "loss": nn.SmoothL1Loss(),
    "feat_loss": nn.HuberLoss(),
    # "extra_head"
}

embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": latent_dim,
    "embedders_config": [
        (
            MultiTaskEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": None,
                "output_dim": latent_dim,
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
    "estimator_params": standard_lgbm_cite_conf,
    "embedder_params": embedder_params,
    "seed": seed,
}
