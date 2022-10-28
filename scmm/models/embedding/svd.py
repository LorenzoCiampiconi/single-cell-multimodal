import logging
import pathlib

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from scmm.models.embedding.base import Embedder

from scmm.utils.caching import caching_method

logger = logging.getLogger(__name__)


class TruncatedSVDEmbedder(Embedder):
    def __init__(self, *, seed: int, input_dim: int, output_dim: int, **kwargs):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.svd = TruncatedSVD(n_components=output_dim, random_state=seed, **kwargs)
        self.fitted = False

    def _load_cached_svd(self, path:pathlib.Path):
        loaded_obj: TruncatedSVDEmbedder = joblib.load(path)

        assert self.input_dim == loaded_obj.input_dim
        assert self.output_dim == loaded_obj.output_dim
        assert self.seed == loaded_obj.seed
        assert isinstance(loaded_obj.svd, TruncatedSVD)

        self.svd = loaded_obj.svd

    @caching_method(
        file_label="embedder",
        file_extension="t-svd",
        loading_method_ref='_load_cached_svd',
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
