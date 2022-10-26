import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from scmm.models.embedding.base import Embedder

logger = logging.getLogger(__name__)


class MultiLevelEmbedder(Embedder):
    def __init__(
        self,
        *,
        seed: int,
        input_dim: int,
        output_dim: int,
        embedders_config: List[Tuple[Type[Embedder], Dict[str, Any]]],
    ):
        super().__init__(seed=seed, input_dim=input_dim, output_dim=output_dim)
        self.embedders_config = embedders_config
        # self.embedders = OrderedDict(((embedder, embedder(**kwargs)) for embedder, kwargs in embedders_config))
        self._fitted_embedder = []
        self.fitted = False

    def transform(self, *, input) -> np.array:
        assert self.fitted and all(
            e.is_fit for e in self._fitted_embedder
        ), "this multi level embedder has not been fitted"
        for embedder in self._fitted_embedder:
            input = embedder.transform(input=input)

        return input

    def fit(self, *, input):
        self._fitted_embedder = []
        for embedder, kwargs in self.embedders_config:
            embedder = embedder(**kwargs).fit(input=input)
            self._fitted_embedder.append(embedder)

        self.fitted = True

    def fit_transform(self, *, input) -> np.array:
        self._fitted_embedder = []
        for embedder, kwargs in self.embedders_config:
            input, embedder = embedder(**kwargs).fit_transform(input=input)
            self._fitted_embedder.append(embedder)

        self.fitted = True

        return input
