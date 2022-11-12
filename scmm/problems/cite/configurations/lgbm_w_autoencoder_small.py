from scmm.models.embedding.autoencoder import BasicAutoEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.common_conf import (
    standard_lgbm_cite_conf,
    cv_params,
    dataloader_kwargs,
    trainer_kwargs,
    standard_autoencoder_net_params,
)
from torch import nn

from scmm.problems.cite.configurations.utils import check_nn_embedder_params

model_label = "lgbm_w_autoencoder"
model_class = LGBMwMultilevelEmbedderCite
seed = 0
original_dim = None

estimator_params = standard_lgbm_cite_conf

logger_kwargs = {
    "name": "basic_autoencoder",
}

latent_dim = 4

net_params = {
    "lr": 1e-3,
    "shrinking_factors": (4, 2),
    "activation_function": nn.SELU,
    "loss": nn.SmoothL1Loss(),
}

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
                "output_dim": 32,
            },
        ),
        (
            BasicAutoEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": 32,
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
    "estimator_params": estimator_params,
    "embedder_params": embedder_params,
    "seed": seed,
}
