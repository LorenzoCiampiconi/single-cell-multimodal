from single_cell_multimodal_core.models.dimensionality_reduction.embedding import AutoEncoderDimensionalityReductionMixin
from single_cell_multimodal_core.models.dimensionality_reduction.svd import TruncatedSVDMixin

model_label = "lgbm_w_autoencoder"

global_params = {
    "seed": 0,
}

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

dimensionality_reduction = {
    "output_dimensionality": 16,
    "multi_level_dimensionality": True,
    "dimensionality_reducers_class": [TruncatedSVDMixin, AutoEncoderDimensionalityReductionMixin],
    "intermediate_dimensionalities": [64]
}

configuration = {
    "cross_validation_params": cross_validation_params,
    "model_params": model_params,
    "dimensionality_reduction_params": dimensionality_reduction,
    "global_params": global_params,
    "intermediate_dim": 2000 #todo
}