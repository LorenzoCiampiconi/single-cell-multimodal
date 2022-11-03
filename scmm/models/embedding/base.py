import abc
import logging
from typing import Type

import numpy as np

logger = logging.getLogger(__name__)


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
    def fit(self, *, input, **kwargs):
        pass

    @abc.abstractmethod
    def transform(self, *, input, **kwargs) -> np.array:
        pass

    def fit_transform(self, *, input, **kwargs) -> np.array:
        logger.info(
            f"{self.__class__.__name__} is being fit with input_dim={self.input_dim} and latent_dim={self.output_dim}"
        )
        self.fit(input=input, **kwargs)
        logger.info("Embedder has been fit - Done")
        self.fitted = True

        if "use_cache" not in kwargs:
            kwargs["use_cache"] = False

        return self.transform(input=input, **kwargs), self

    def inverse_transform(self, *, input, **kwargs):
        raise NotImplementedError(
            f"{self.__class__} does not implement any inverse function for the mapping to a space with a smaller dimensionality"
        )


class EmbedderWrapperInputMixin(metaclass=abc.ABCMeta):
    configuration: dict

    @property
    @abc.abstractmethod
    def embedder_class(self) -> Type[Embedder]:
        pass

    @property
    def embedder_kwargs(self) -> dict:
        return self.configuration["embedder_params"]

    def fit_and_apply_dimensionality_reduction(self, *, input, **kwargs):
        self.embedder = self.embedder_class(**self.embedder_kwargs)
        out, self.embedder = self.embedder.fit_transform(input=input, **kwargs)
        return out

    def apply_dimensionality_reduction(self, input, **kwargs):
        return self.embedder.transform(input=input, **kwargs)


class EmbedderWrapperOutputMixin(metaclass=abc.ABCMeta):
    configuration: dict

    @property
    @abc.abstractmethod
    def embedder_output_class(self) -> Type[Embedder]:
        pass

    @property
    def embedder_output_kwargs(self) -> dict:
        return self.configuration["embedder_output_params"]

    def fit_and_apply_dimensionality_reduction_to_target(self, *, input, **kwargs):
        self.embedder = self.embedder_output_class(**self.embedder_output_kwargs)
        out, self.embedder = self.embedder.fit_transform(input=input, **kwargs)
        return out

    def apply_dimensionality_reduction_to_target(self, input, **kwargs):
        return self.embedder.transform(input=input, **kwargs)

    def apply_dimensionality_expansion_to_prediction(self, input, **kwargs):
        return self.embedder.transform(input=input, **kwargs)
