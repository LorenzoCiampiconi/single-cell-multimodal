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
        self.embedders = OrderedDict(((embedder, embedder(**kwargs)) for embedder, kwargs in embedders_config))
        self.fitted = False

    def transform(self, *, input) -> np.array:
        for embedder in self.embedders.values():
            input = embedder.transform(input=input)

        return input

    def fit(self, *, input):
        self.fit_transform(input=input)
        self.fitted = True

    def fit_transform(self, *, input) -> np.array:
        for embedder in self.embedders.values():
            input = embedder.fit_transform(input=input)
        self.fitted = True

        return input
