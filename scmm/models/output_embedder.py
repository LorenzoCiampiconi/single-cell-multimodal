import abc
from typing import Type

from sklearn.base import BaseEstimator

from scmm.models.embedding.base_embedder import Embedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder


class FittablePredictingEstimator(BaseEstimator):
    @abc.abstractmethod
    def fit(self, X, Y, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        pass


class OutputDimensionalityReducedModelWrapper(FittablePredictingEstimator):
    def __init__(self, *, wrapped_model, embedder_class, embedder_params):
        self._is_fitted = False
        self._wrapped_model = wrapped_model
        self._dimensionality_reducer: Embedder = embedder_class(**embedder_params)

    def transform_label(self, Y, **kwargs):
        if self._dimensionality_reducer.is_fit:
            return self._dimensionality_reducer.transform(Y=Y, **kwargs)
        else:
            Y_r, self._dimensionality_reducer = self._dimensionality_reducer.fit_transform(
                input=Y, runtime_labelling=f"odr", **kwargs
            )
            return Y_r

    def inverse_transform_label(self, Y, **kwargs):
        assert self._dimensionality_reducer.is_fit
        return self._dimensionality_reducer.inverse_transform(Y=Y, **kwargs)

    def fit(self, X, Y, **kwargs) -> FittablePredictingEstimator:
        Y_r, self._dimensionality_reducer = self._dimensionality_reducer.fit_transform(
            input=Y, runtime_labelling=f"odr", **kwargs
        )
        self._wrapped_model.fit(X, Y_r)
        self._is_fitted = True
        return self

    def predict(self, X, **kwargs):
        assert self._is_fitted, f"{self} has not been fitted yet."
        Y_hat_r = self._wrapped_model.predict(X, **kwargs)
        Y_hat = self._dimensionality_reducer.inverse_transform(input=Y_hat_r, **kwargs)
        return Y_hat

    def fit_predict(self, X, Y, **kwargs):
        return self.fit(X, Y, **kwargs).predict(X)


class ODRModelWrappedMixin(metaclass=abc.ABCMeta):
    """
    ODR stands for Output Dimensionality Reduced
    """

    model_class: Type
    estimator_params: dict
    model_instantiation_kwargs: dict
    odr_embedder_class: Type[Embedder]
    configuration: dict

    @property
    def odr_embedder_params(self):
        return self.configuration["odr_params"]["embedder_params"]

    @property
    @abc.abstractmethod
    def odr_embedder_class(self):
        pass

    @property
    def perform_odr_also_on_target_for_input_dim_reduction(self):
        return (
            "odr_params" in self.configuration
            and "odr_also_on_target_for_input_dim_reduction" in self.configuration["odr_params"]
            and self.configuration["odr_params"]["odr_also_on_target_for_input_dim_reduction"]
        )

    def pre_process_target_for_dim_reduction(self, Y, model_kwargs=None):
        if self.perform_odr_also_on_target_for_input_dim_reduction:
            if model_kwargs is None:
                model_kwargs = self.model_instantiation_kwargs
            odr_obj: OutputDimensionalityReducedModelWrapper = self.instantiate_estimator(**model_kwargs)
            Y_r = odr_obj.transform_label(Y)
            return Y_r
        else:
            return Y

    def instantiate_estimator(self, **model_instantiation_kwargs):
        return OutputDimensionalityReducedModelWrapper(
            wrapped_model=super().instantiate_estimator(**model_instantiation_kwargs),
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
