model_label = "lgbm_w_svd_baseline"

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

svd_params = {"output_dimensionality": 64}

configuration = {
    "cross_validation_params": cross_validation_params,
    "model_params": model_params,
    "dimensionality_reduction_params": svd_params,
    "global_params": global_params,
}
