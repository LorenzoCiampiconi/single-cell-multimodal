from scmm.models.embedding.autoencoder import AutoEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder

model_label = "lgbm_w_autoencoder"
seed = 0

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

embedder_params = {
    "seed": seed,
    "input_dim": 20000,
    "output_dim": 16,
    "embedders_config": [
        (
            TruncatedSVDEmbedder,
            {
                "seed": seed,
                "input_dim": 20000,
                "output_dim": 2000,
            },
        ),
        (
            AutoEncoderEmbedder,
            {
                "seed": seed,
                "input_dim": 2000,
                "output_dim": 64,
                "model_kwargs": {},
                "train_kwargs": {},
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
