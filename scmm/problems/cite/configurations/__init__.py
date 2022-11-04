from . import (
    lgbm_w_autoencoder_small,
    lgbm_w_autoencoder_deep,
    lgbm_w_autoencoder_deep,
    lgbm_w_supervised_autoencoder_deep,
    lgbm_w_svd
)

config_dict = {
    "lgbm_w_svd": lgbm_w_svd,
    "small": lgbm_w_autoencoder_small,
    "deep": lgbm_w_autoencoder_deep,
    "supervised_small": lgbm_w_autoencoder_deep,
    "supervised_deep": lgbm_w_supervised_autoencoder_deep,
}
