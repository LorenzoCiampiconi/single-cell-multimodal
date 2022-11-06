import abc
from typing import Dict, Type

from sklearn.multioutput import MultiOutputRegressor


class MultiModelWrapperMixin(metaclass=abc.ABCMeta):
    model_params: Dict
    model_class: Type

    @property
    @abc.abstractmethod
    def model_wrapper_class(self):
        pass

    def instantiate_model(self, **model_instantiation_kwargs):
        return self.model_wrapper_class(**model_instantiation_kwargs)

    @property
    def model_instantiation_kwargs(self):
        return {"estimator": self.model_class(**self.model_params)}

    def build_model_params_for_tuning(self, params):
        return {"estimator": self.model_class(**params)}


class MultiOutputRegressorMixin(MultiModelWrapperMixin):
    @property
    def model_wrapper_class(self):
        return MultiOutputRegressor
