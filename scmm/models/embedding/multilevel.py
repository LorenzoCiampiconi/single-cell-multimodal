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
        self._fitted_embedder: List[Embedder] = []

    def transform(self, *, input, **kwargs) -> np.array:
        assert self.is_fit and all(
            e.is_fit for e in self._fitted_embedder
        ), "this multi level embedder has not been fitted"
        for embedder in self._fitted_embedder:
            input = embedder.transform(input=input, **kwargs)

        return input

    def fit(self, *, input, **kwargs):
        self._fitted_embedder = []
        for embedder, init_kwargs in self.embedders_config:
            input, embedder = embedder(**init_kwargs).fit_transform(input=input, **kwargs)
            self._fitted_embedder.append(embedder)
            # runtime_labelling = kwargs['runtime_labelling']
            # runtime_labelling += "" #todo

        self._fitted = True

        return self

    def inverse_transform(self, *, input, **kwargs):
        assert self.is_fit and all(
            e.is_fit for e in self._fitted_embedder
        ), "this multi level embedder has not been fitted"
        for embedder in reversed(self._fitted_embedder):
            input = embedder.inverse_transform(input=input)

        return input
