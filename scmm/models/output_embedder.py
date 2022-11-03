import abc
from typing import Type

from sklearn.base import BaseEstimator

from scmm.models.embedding.base import Embedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder


class FittablePredictingEstimator(BaseEstimator):
    @abc.abstractmethod
    def fit(self, X, Y, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


class OutputDimensionalityReducedModelWrapper(FittablePredictingEstimator):
    def __init__(self, *, wrapped_model, embedder_class, embedder_params):
        self._is_fitted = False
        self._wrapped_model = wrapped_model
        self._dimensionality_reducer: Embedder = embedder_class(**embedder_params)

    def fit(self, X, Y, **kwargs) -> FittablePredictingEstimator:
        Y_r, self._dimensionality_reducer = self._dimensionality_reducer.fit_transform(input=Y, runtime_labelling=f"odr", **kwargs)
        self._wrapped_model.fit(X, Y_r)
        self._is_fitted = True
        return self

    def predict(self, X):
        assert self._is_fitted, f"{self} has not been fitted yet."
        Y_hat_r = self._wrapped_model.predict(X)
        Y_hat = self._dimensionality_reducer.inverse_transform(Y_hat_r)
        return Y_hat

    def fit_predict(self, X, Y, **kwargs):
        return self.fit(X, Y, **kwargs).predict(X)


class ODRModelWrappedMixin(metaclass=abc.ABCMeta):
    """
    ODR stands for Output Dimensionality Reduced
    """

    model_class: Type
    model_params: dict
    odr_embedder_class: Type[Embedder]
    configuration: dict

    @property
    def odr_embedder_params(self):
        return self.configuration["odr_params"]["embedder_params"]

    @property
    @abc.abstractmethod
    def odr_embedder_class(self):
        pass

    def instantiate_model(self, **model_instantiation_kwargs):
        return OutputDimensionalityReducedModelWrapper(
            wrapped_model=super().instantiate_model(**model_instantiation_kwargs),
            embedder_class=self.odr_embedder_class,
            embedder_params=self.odr_embedder_params,
        )


class ODRwSVDMixin(ODRModelWrappedMixin):
    """
    ODR stands for Output Dimensionality Reduced
    """

    @property
    def odr_embedder_class(self):
        return TruncatedSVDEmbedder
