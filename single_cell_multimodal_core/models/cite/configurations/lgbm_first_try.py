model_label = 'lgbm_first_try'

params = {
     'learning_rate': 0.1,
     'objective' : 'regression',
     'metric': 'rmse',#mae',
     'random_state': 4223,
     'reg_alpha': 0.03,
     'reg_lambda': 0.002,
     'colsample_bytree': 0.8,
     'subsample': 0.6,
     'max_depth': 10,
     'num_leaves': 186,
     'min_child_samples': 263
    }

cross_validation_params = {
     'n_splits_for_kfold':2
}