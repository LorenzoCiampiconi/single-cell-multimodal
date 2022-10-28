from scmm.models.embedding.autoencoder import BasicAutoEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from torch import nn

model_label = "lgbm_w_deep_autoencoder"
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

cross_validation_params = {"n_splits_for_kfold": 10}

logger_kwargs = {
    "name": "basic_autoencoder",
}
dataloader_kwargs = {
    "batch_size": 64,
    "num_workers": 0,
}
trainer_kwargs = {
    # "accelerator": "gpu",
    "max_epochs": 50,
    "check_val_every_n_epoch": 1,
    # "val_check_interval": 1,
    "log_every_n_steps": 50,
    "gradient_clip_val": 1,
}
net_params = {
    "lr": 1e-3,
    "shrinking_factors": (2, 2, 2, 2),
    "activation_function": nn.SELU,
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
            BasicAutoEncoderEmbedder,
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

for embedder_config in embedder_params["embedders_config"]:
    if "model_params" in embedder_config[1] and "shrinking_factors" in embedder_config[1]["model_params"]:
        input_dim = embedder_config[1]["input_dim"]
        output_dim = embedder_config[1]["output_dim"]

        final_dim = input_dim
        for factor in embedder_config[1]["model_params"]["shrinking_factors"]:
            final_dim = final_dim // factor

        assert final_dim == output_dim


configuration = {
    "cross_validation_params": cross_validation_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "seed": seed,
}

model_label = f"lgbm_w_{len(net_params['shrinking_factors'])}lrs-deep_autoencoder_dim-{svd_out_dim}->{latent_dim}"
