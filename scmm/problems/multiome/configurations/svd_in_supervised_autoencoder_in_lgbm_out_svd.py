import torchmetrics
from torch import nn

from scmm.models.embedding.autoencoder.full.concrete.multitask import MultiTaskEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from scmm.problems.cite.configurations.utils import check_nn_embedder_params
from scmm.problems.metrics import common_metrics
from scmm.problems.multiome.concrete import SVDwAutoencoderInLGBMOutSVDMultiome

model_label = "svd_in_lgbm_out_svd"
model_class = SVDwAutoencoderInLGBMOutSVDMultiome
seed = 0
original_dim = None

cv_params = {"cv": 3, "scoring": common_metrics, "verbose": 10}

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

output_reduced_dim = 64

output_embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": output_reduced_dim,
}

odr_params = {"embedder_params": output_embedder_params, "odr_also_on_target_for_input_dim_reduction": True}

logger_kwargs = {
    "name": "supervised_autoencoder_cite",
}
dataloader_kwargs = {
    "batch_size": 64,
    "num_workers": 4,
}
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
    "shrinking_factors": (2, 2, 2, 2, 2),
    "activation_function": nn.SELU,
    "input_coef": 1 / 7,
    "features_dim": output_reduced_dim,
    "loss": nn.SmoothL1Loss(),
    "feat_loss": nn.HuberLoss(),
    # "extra_head"
}

svd_out_dim = 4096
latent_dim = 128

embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": latent_dim,
    "embedders_config": [
        (
            TruncatedSVDEmbedder,
            {
                "seed": seed,
                "input_dim": original_dim,
                "output_dim": svd_out_dim,
            },
        ),
        (
            MultiTaskEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": svd_out_dim,
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

check_nn_embedder_params(embedder_params)

configuration = {
    "cv_params": cv_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "odr_params": odr_params,
    "seed": seed,
}
