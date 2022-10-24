import abc
import logging
from typing import Union, Type

import numpy as np

from single_cell_multimodal_core.models.dimensionality_reduction.base import DimensionalityReductionMixinABC
from single_cell_multimodal_core.models.embedding.base import Embedder
from single_cell_multimodal_core.models.embedding.concrete import AutoEncoderEmbedder

logger = logging.getLogger(__name__)

class EmbeddingDimensionalityReductionMixinABC(DimensionalityReductionMixinABC,metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedder: Union[Embedder, None] = None

    @property
    @abc.abstractmethod
    def embedder_class(self) -> Type[Embedder]:
        pass

    @property
    @abc.abstractmethod
    def embedder_kwargs(self) -> dict:
        pass

    def apply_dimensionality_reduction(self, *, input) -> np.array:
        return self._embedder.transform(input)

    def fit_and_apply_dimensionality_reduction(self, *, input, output_dimensionality=64) -> np.array: #todo clean method argument
        logger.info(f"Istantiating the embedder {self.embedder_class} with input_dim={input.shape[1]} and latent_dim={output_dimensionality}")

        if self._embedder is None:
            self._embedder = self.embedder_class(input_dim=input.shape[1], latent_dim=output_dimensionality, **self.embedder_kwargs)

        logger.info(f"Fitting the embedder {self.embedder_class} to an input of dimension {input.shape[1]}")
        return self._embedder.fit_transform(input)


class AutoEncoderMixin:
    seed: int

    @property
    def embedder_class(self) -> Type[Embedder]:
        return AutoEncoderEmbedder

    @property
    def embedder_kwargs(self) -> dict:
        return {"seed": self.seed}

class AutoEncoderDimensionalityReductionMixin(
    AutoEncoderMixin,
    EmbeddingDimensionalityReductionMixinABC
):
    ...
