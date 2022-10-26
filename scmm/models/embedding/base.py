import abc
import logging
from typing import Type

import numpy as np

logger = logging.getLogger()


class Embedder(metaclass=abc.ABCMeta):
    is_fit: bool

    def __init__(self, *, seed: int, input_dim: int, output_dim: int):
        self.seed = seed
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def is_fit(self) -> bool:
        return self.fitted

    @abc.abstractmethod
    def fit(self, *, input):
        pass

    @abc.abstractmethod
    def transform(self, *, input) -> np.array:
        pass

    def fit_transform(self, *, input) -> np.array:
        logger.info(
            f"{self.__class__.__name__} is being fit with input_dim={self.input_dim} and latent_dim={self.output_dim}"
        )
        fitted_embedder = self.fit(input=input)
        self.fitted = True
        logger.info("Embedder has been fit - Done")

        logger.info("Now transforming the input")
        return fitted_embedder.transform(input=input), fitted_embedder


class EmbedderWrapperMixin(metaclass=abc.ABCMeta):
    configuration: dict

    @property
    @abc.abstractmethod
    def embedder_class(self) -> Type[Embedder]:
        pass

    @property
    def embedder_kwargs(self) -> dict:
        return self.configuration["embedder_params"]

    def fit_and_apply_dimensionality_reduction(self, *, input):
        self.embedder = self.embedder_class(**self.embedder_kwargs)
        out, self.embedder = self.embedder.fit_transform(input=input)
        return out

    def apply_dimensionality_reduction(self, input):
        return self.embedder.transform(input=input)
