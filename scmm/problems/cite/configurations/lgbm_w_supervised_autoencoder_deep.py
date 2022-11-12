from scmm.models.embedding.autoencoder.full.concrete.multitask import MultiTaskEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.common_conf import standard_lgbm_cite_conf, cv_params
from scmm.problems.cite.configurations.utils import check_nn_embedder_params
from scmm.problems.metrics import common_metrics
from torch import nn

model_class = LGBMwMultilevelEmbedderCite
seed = 0
original_dim = None

estimator_params = standard_lgbm_cite_conf

logger_kwargs = {
    "name": "supervised_autoencoder_cite",
}

dataloader_kwargs = {
    "batch_size": 128,
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
    "shrinking_factors": (2, 2, 2, 2),
    "activation_function": nn.SELU,
    "input_coef": 1 / 10,
    "features_dim": 140,
    # "extra_head"
}

svd_out_dim = 2048
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
                "estimator_params": net_params,
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

model_label = (
    f"lgbm_w_{len(net_params['shrinking_factors'])}lrs-deep_supervised_autoencoder_dim-{svd_out_dim}->{latent_dim}"
)
