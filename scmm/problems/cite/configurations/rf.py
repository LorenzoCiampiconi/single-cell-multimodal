from scmm.models.embedding.autoencoder import BasicAutoEncoderEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder
from torch import nn

model_label = "rf"
seed = 0
original_dim = None

model_params = {}

cross_validation_params = {"n_splits_for_kfold": 10}


embedder_params = {
    "seed": seed,
    "input_dim": original_dim,
    "output_dim": original_dim,
}

configuration = {
    "cross_validation_params": cross_validation_params,
    "model_params": model_params,
    "embedder_params": embedder_params,
    "seed": seed,
}
