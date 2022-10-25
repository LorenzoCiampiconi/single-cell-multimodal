import logging
from scmm.models.embedding.base import Embedder
import numpy as np

logger = logging.getLogger(__name__)

class NOPEmbedder(Embedder):
    def __init__(self, *, seed: int, input_dim: int, output_dim: int, **kwargs):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.fitted = False

    def fit(self, *, input):
        self.fitted = True

    def transform(self, *, input) -> np.array:
        return input
