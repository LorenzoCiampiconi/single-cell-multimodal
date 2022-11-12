from copy import deepcopy

import pandas as pd

from scmm.problems.cite.concrete import EnsembleSplitTargetSVDCite, LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.common_conf import cv_params, original_dim
from scmm.problems.cite.configurations.ensemble.lgbm_w_autoencoder_no_svd import configuration, model_class
from scmm.utils.appdirs import app_static_dir

sub_model_class = model_class

model_class = EnsembleSplitTargetSVDCite
model_label = "ensemble_split_target_cite"
seed = 0

common_level_svd_output_dim = 2048

embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": common_level_svd_output_dim,
}

df_split = pd.read_csv(app_static_dir("cache/clustering") / "baseline_split_cite.csv")
group_1 = df_split[df_split.group == 0]["target_id"].values
group_2 = df_split[df_split.group == 1]["target_id"].values

subsets_mapping = {
    tuple(range(0,70)): {"configuration": deepcopy(configuration), "model_class": sub_model_class},
    tuple(range(70,140)): {
        "configuration": deepcopy(configuration),
        "model_class": sub_model_class,
    },
}

for subset in subsets_mapping:
    subsets_mapping[subset]["configuration"]["embedder_params"]["input_dim"] = common_level_svd_output_dim
    subsets_mapping[subset]["configuration"]["embedder_params"]["embedders_config"][0][1][
        "input_dim"
    ] = common_level_svd_output_dim
    subsets_mapping[subset]["configuration"]["embedder_params"]["embedders_config"][0][1]["model_params"][
        "features_dim"
    ] = len(subset)

estimator_params = {"subset_mapping": subsets_mapping, "output_dim": 140}

configuration = {
    "estimator_params": estimator_params,
    "cv_params": cv_params,
    "embedder_params": embedder_params,
    "seed": seed,
}
