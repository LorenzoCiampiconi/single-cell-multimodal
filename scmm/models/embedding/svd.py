import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD

from scmm.models.embedding.base import Embedder

# from single_cell_multimodal_core.utils.caching import caching_function

logger = logging.getLogger(__name__)


class TruncatedSVDEmbedder(Embedder):
    def __init__(self, *, seed: int, input_dim: int, output_dim: int, **kwargs):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.svd = TruncatedSVD(n_components=output_dim, random_state=seed, **kwargs)
        self.fitted = False

    def fit(self, *, input):
        self.svd.fit(input)
        self.fitted = True

    def transform(self, *, input) -> np.array:
        return self.svd.transform(input=input)
