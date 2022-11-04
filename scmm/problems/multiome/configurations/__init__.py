from . import svd_ridge, svd_in_lgbm_out_svd, ridge_w_autoencoder_deep

config_dict = {
    "small": svd_ridge,
    "svd_in_lgbm_out_svd": svd_in_lgbm_out_svd,
    "ridge_w_autoencoder_deep": ridge_w_autoencoder_deep,
}
