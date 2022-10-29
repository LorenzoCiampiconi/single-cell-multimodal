from scmm.problems.metrics import common_metrics

model_label = "ridge_w_svd_baseline"
seed = 0
original_dim = None


model_params = {
    "copy_X": False
}

cv_params = {"cv": 3, "scoring": common_metrics, "verbose": 10}

svd_params = {"output_dimensionality": 4096}

embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": 4096,
}

configuration = {
    "cv_params": cv_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "seed": seed,
}