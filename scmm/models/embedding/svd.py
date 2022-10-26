import logging

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from scmm.models.embedding.base import Embedder

from scmm.utils.caching import caching_function

logger = logging.getLogger(__name__)


def _load_svd_model(self, file):
    return joblib.load(file)


class TruncatedSVDEmbedder(Embedder):
    def __init__(self, *, seed: int, input_dim: int, output_dim: int, **kwargs):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.svd = TruncatedSVD(n_components=output_dim, random_state=seed, **kwargs)
        self.fitted = False

    @caching_function(
        file_label="embedder",
        file_extension="t-svd",
        loading_function=joblib.load,
        saving_function=joblib.dump,
        labelling_kwargs={},
        object_labelling_attributes=("input_dim", "output_dim", "seed"),
        cache_folder="svd",
    )  # todo improve when will cache also multiome dimensionality reduction
    def fit(self, *, input):
        self.svd.fit(input)
        self.fitted = True
        return self

    def transform(self, *, input) -> np.array:
        return self.svd.transform(input)
