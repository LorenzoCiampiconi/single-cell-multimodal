import abc
import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import sparse
from scmm.utils.appdirs import app_static_dir
from scmm.utils.scikit_crossval import cross_validate
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


class SCMModelABC(metaclass=abc.ABCMeta):
    invalid_test_index: int = 7476
    public_test_index: int

    def __init__(self, configuration: Dict[str, Dict[str, Any]], label=""):
        self._configuration = configuration
        self._model_label = f"{label}" if label else self.__class__.__name__
        self._trained_model = None

    @property
    def model_label(self):
        return f"{self.problem_label}_{self._model_label}"

    @property
    def is_trained(self):
        return self._trained_model is not None

    @property
    @abc.abstractmethod
    def problem_label(self) -> str:
        pass

    @property
    def configuration(self):
        return self._configuration

    @property
    @abc.abstractmethod
    def train_input(self) -> sparse.csr_array:
        pass

    @property
    @abc.abstractmethod
    def train_target(self) -> sparse.csr_array:
        pass

    @property
    @abc.abstractmethod
    def test_input(self) -> sparse.csr_array:
        pass

    @abc.abstractmethod
    def instantiate_model(self, **model_instantiation_kwargs):
        pass

    @property
    @abc.abstractmethod
    def model_instantiation_kwargs(self):
        pass

    @property
    def cv_params(self):
        return self.configuration["cv_params"]

    @property
    def model_params(self):
        return self.configuration["model_params"]

    @property
    def seed(self):
        return self.configuration["seed"]

    @property
    def embedder_params(self):
        return self.configuration["embedder_params"]

    def fit_and_apply_dimensionality_reduction(self, input):
        return input

    def apply_dimensionality_reduction(self, input):
        return input

    def cross_validation(self, X, Y, **kwargs):
        # TODO with strategy: {self.cv_params['strategy']}")
        instantiate_model = lambda: self.instantiate_model(**self.model_instantiation_kwargs)
        cv_raw = cross_validate(instantiate_model, X, Y, **self.cv_params)
        cv_out = self.process_cv_out(cv_raw)
        return cv_out

    def process_cv_out(self, cv_raw):
        return pd.DataFrame(cv_raw)

    def fit_model(self, X, Y):
        self._trained_model = self.instantiate_model(**self.model_instantiation_kwargs).fit(X, Y)

    def full_pipeline(self, refit=True):
        X, Y = self.train_input, self.train_target

        logger.debug(f"{self.model_label} - applying dimensionality reduction")
        X = self.fit_and_apply_dimensionality_reduction(input=X)
        logger.debug(f"{self.model_label} - applying dimensionality reduction - Done")

        logger.info(f"{self.model_label} - performing cross validation")
        cv_out = self.cross_validation(X, Y)
        logger.info(f"{self.model_label} - Average  metrics: " + " | ".join([f"({k})={v:.4}" for k, v in cv_out.mean().items()]))

        if refit:
            self.fit_model(X, Y)
            Y_test = self.predict_public_test()
            self.generate_public_test_output(Y_test)

        return cv_out

    def predict_public_test(self) -> np.array:
        X_test_reduced = self.apply_dimensionality_reduction(input=self.test_input)
        return self._trained_model.predict(X_test_reduced)

    def generate_public_test_output(self, test_output: np.array):
        test_output[:7476] = 0
        submission = pd.read_csv(app_static_dir("data") / "sample_submission.csv", index_col="row_id").squeeze(
            "columns"
        )
        submission.iloc[self.public_test_index : len(test_output.ravel())] = test_output.ravel()
        assert not submission.isna().any()
        submission.to_csv(
            app_static_dir("out") / f"{self.model_label}_{datetime.now().strftime('%Y%m%d-%H%M')}_submission.csv"
        )


class MultiModelWrapperMixin(metaclass=abc.ABCMeta):
    model_params: Dict

    @property
    @abc.abstractmethod
    def model_wrapper_class(self):
        pass

    @property
    @abc.abstractmethod
    def model_class(self):
        pass

    def instantiate_model(self, **model_instantiation_kwargs):
        return self.model_wrapper_class(**model_instantiation_kwargs)

    @property
    def model_instantiation_kwargs(self):
        return {"estimator": self.model_class(**self.model_params)}


class MultiOutputRegressorMixin(MultiModelWrapperMixin):
    @property
    def model_wrapper_class(self):
        return MultiOutputRegressor
