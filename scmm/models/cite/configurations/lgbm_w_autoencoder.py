from scmm.models.embedding.autoencoder import BasicAutoEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from torch import nn

model_label = "lgbm_w_autoencoder"
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
    "max_epochs": 1,
    "check_val_every_n_epoch": 1,
    # "val_check_interval": 1,
    "log_every_n_steps": 50,
    "gradient_clip_val": 1,
}
net_params = {
    "lr": 1e-3,
    "shrinking_factors": (8, 2),
    "activation_function": nn.SELU,
}
embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": 4,
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
            BasicAutoEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": 64,
                "output_dim": 4,
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
    "cross_validation_params": cross_validation_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "seed": seed,
}
