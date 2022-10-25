import abc
from datetime import datetime
import gc
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from scmm.utils.metrics import correlation_score
from scmm.utils.appdirs import app_static_dir

logger = logging.getLogger(__name__)


class SCMModelABC(metaclass=abc.ABCMeta):
    invalid_test_index = 7476

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
    def cross_validation_params(self):
        return self.configuration["cross_validation_params"]

    @property
    def model_params(self):
        return self.configuration["model_params"]

    @property
    def seed(self):
        return self.configuration["seed"]

    @property
    def embedder_params(self):
        return self.configuration["embedder_params"]

    @abc.abstractmethod
    def fit_and_apply_dimensionality_reduction(self, input):
        ...

    @abc.abstractmethod
    def apply_dimensionality_reduction(self, input):
        ...

    def cross_validation(self, X, Y, save_model=False):
        logger.info(f"{self.model_label} - performing cross validation")

        kf = KFold(n_splits=self.cross_validation_params["n_splits_for_kfold"], shuffle=True, random_state=1)
        score_list = []

        for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
            model = None
            gc.collect()
            X_tr = X[idx_tr]
            y_tr = Y[idx_tr]

            model = self.instantiate_model(**self.model_instantiation_kwargs)
            model.fit(X_tr, y_tr)
            del X_tr, y_tr
            gc.collect()

            # We validate the model_wrapper
            X_va = X[idx_va]
            y_va = Y[idx_va]

            y_va_pred = model.predict(X_va)
            # va_pred.append(y_va_pred)
            mse = mean_squared_error(y_va, y_va_pred)
            corr_score = correlation_score(y_va, y_va_pred)
            del X_va, y_va

            logger.info(f"{self.model_label} - Fold {fold}: mse = {mse:.5f}, corr =  {corr_score:.3f}")
            score_list.append((mse, corr_score))

        result_df = pd.DataFrame(score_list, columns=["mse", "corrscore"])
        logger.info(
            f"{self.model_label} - Average  mse = {result_df.mse.mean():.5f}; corr = {result_df.corrscore.mean():.3f}"
        )

        self._trained_model = self.instantiate_model(**self.model_instantiation_kwargs).fit(X, Y)

        if save_model:
            model_path: Path = app_static_dir("saved_models") / f"model_{self.model_label}.pkl"
            model_path.write_bytes(pkl.dumps(self._trained_model))

        return result_df

    def full_pipeline(self, save_model=False):
        X, Y = self.train_input, self.train_target

        logger.debug(f"{self.model_label} - applying dimensionality reduction")
        X = self.fit_and_apply_dimensionality_reduction(input=X)
        logger.debug(f"{self.model_label} - applying dimensionality reduction - Done")

        cv_results = self.cross_validation(X, Y, save_model=save_model)

        Y_test = self.predict_public_test()

        self.generate_public_test_output(Y_test)

    def predict_public_test(self) -> np.array:
        X_test_reduced = self.apply_dimensionality_reduction(input=self.test_input)
        return self._trained_model.predict(X_test_reduced)

    def generate_public_test_output(self, test_output: np.array):
        test_output[:7476] = 0
        submission = pd.read_csv(app_static_dir("data") / "sample_submission.csv", index_col="row_id").squeeze("columns")
        submission.iloc[: len(test_output.ravel())] = test_output.ravel()
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
