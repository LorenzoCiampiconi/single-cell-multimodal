import abc
from typing import Dict, Type

from sklearn.multioutput import MultiOutputRegressor


class MultiOutputEstimatorWrapperMixin(metaclass=abc.ABCMeta):
    estimator_params: Dict
    model_class: Type

    @property
    @abc.abstractmethod
    def estimator_wrapper_class(self):
        pass

    def instantiate_estimator(self, **model_instantiation_kwargs):
        return self.estimator_wrapper_class(**model_instantiation_kwargs)

    @property
    def model_instantiation_kwargs(self):
        return {"estimator": self.estimator_class(**self.estimator_params)}

    def build_estimator_params_for_tuning(self, params):
        return {"estimator": self.estimator_class(**params)}


class MultiOutputRegressorMixin(MultiOutputEstimatorWrapperMixin):
    @property
    def estimator_wrapper_class(self):
        return MultiOutputRegressor

    def _is_estimator_fit(self, estimator):
        return True  # todo
