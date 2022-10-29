from scmm.problems.metrics import common_metrics
from scmm.problems.multiome.concrete import SVDinLGBMoutSVDMultiome

model_label = "svd_in_lgbm_out_svd"
model_class = SVDinLGBMoutSVDMultiome
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

output_embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": 256,
}

odr_params = {
    "embedder_params": output_embedder_params
}

embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": 64,
}

configuration = {
    "cv_params": cv_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "odr_params": odr_params,
    "seed": seed,
}