from scmm.models.embedding.autoencoder import BasicAutoEncoderEmbedder
from scmm.models.embedding.autoencoder.full.concrete.multitask import MultiTaskEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.common_conf import standard_lgbm_cite_conf, \
    dataloader_kwargs, trainer_kwargs, cv_params
from scmm.problems.cite.configurations.lgbm_w_supervised_autoencoder_deep import net_params
from scmm.problems.cite.configurations.utils import check_nn_embedder_params
from scmm.problems.metrics import common_metrics
from torch import nn

model_label = "lgbm_w_supervised_autoencoder_small"
model_class = LGBMwMultilevelEmbedderCite
seed = 0
original_dim = None

model_params = standard_lgbm_cite_conf


logger_kwargs = {
    "name": "supervised_autoencoder_small",
}


latent_dim = 4

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
                "output_dim": 64,
            },
        ),
        (
            MultiTaskEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": 64,
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
    "seed": seed,
}
