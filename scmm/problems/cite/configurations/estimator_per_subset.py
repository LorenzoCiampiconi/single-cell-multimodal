from scmm.problems.cite.concrete import EnsembleSplitTargetSVDCite, LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.common_conf import cv_params, original_dim
from scmm.problems.cite.configurations.ensemble.lgbm_w_autoencoder_no_svd import configuration, model_class

sub_model_class = model_class

model_class = EnsembleSplitTargetSVDCite
model_label = "ensemble_split_target_cite"
seed = 0

common_level_svd_output_dim = 64

embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": common_level_svd_output_dim,
}

subsets_mapping = {
    tuple(range(0, 70)): {
        "configuration": configuration,
        "model_class": sub_model_class
    },
    tuple(range(70, 140)): {
        "configuration": configuration,
        "model_class": sub_model_class,
    }
}

for subset in subsets_mapping:
    subsets_mapping[subset]["configuration"]["embedder_params"]["output_dim"] = common_level_svd_output_dim
    subsets_mapping[subset]["configuration"]["embedder_params"]["embedders_config"][0][1]["input_dim"] = common_level_svd_output_dim
    subsets_mapping[subset]["configuration"]["embedder_params"]["embedders_config"][0][1]["model_params"]["features_dim"] = len(subset)


configuration = {
    "cv_params": cv_params,
    "subsets_mapping": subsets_mapping,
    "embedder_params": embedder_params,
    "seed": seed,
}
