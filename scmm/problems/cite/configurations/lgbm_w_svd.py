from scmm.problems.cite.configurations.common_conf import cv_params, standard_lgbm_cite_conf, original_dim
from scmm.problems.metrics import common_metrics
from scmm.problems.cite.concrete import LGBMwSVDCite

model_label = "lgbm_w_svd_baseline"
model_class = LGBMwSVDCite
seed = 0



embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": 64,
}

configuration = {
    "cv_params": cv_params,
    "model_params": standard_lgbm_cite_conf,
    "embedder_params": embedder_params,
    "seed": seed,
}
