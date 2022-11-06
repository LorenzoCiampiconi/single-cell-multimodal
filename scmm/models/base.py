import abc
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Self

import optuna
import numpy as np
import pandas as pd
from scipy import sparse
from scmm.utils.appdirs import app_static_dir
from scmm.utils.scikit_crossval import cross_validate

logger = logging.getLogger(__name__)


class ModelWrapperABC(metaclass=abc.ABCMeta):
    invalid_test_index: int = 7476
    public_test_index: int

    def __init__(self, configuration: Dict[str, Dict[str, Any]], label=""):
        self._configuration = configuration
        self._model_label = f"{label}" if label else self.__class__.__name__
        self._trained_model = None

    # labels
    @property
    def model_label(self):
        return f"{self.problem_label}_{self._model_label}"

    @property
    @abc.abstractmethod
    def problem_label(self) -> str:
        pass

    # config
    @property
    def configuration(self):
        return self._configuration

    @property
    def seed(self):
        return self.configuration["seed"]

    @property
    @abc.abstractmethod
    def model_class(self):
        pass

    @property
    def model_params(self):
        return self.configuration["model_params"]

    @property
    def cv_params(self):
        return self.configuration["cv_params"]

    # inputs
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

    # build, load & save
    def instantiate_model(self, **kwargs):
        return self.model_class(**kwargs)

    @abc.abstractmethod
    def load(self, path: Path | str) -> Self:
        pass

    @abc.abstractmethod
    def save(self, path: Path | str) -> None:
        pass

    # training methods
    @property
    def is_trained(self):
        return self._trained_model is not None

    def fit_and_apply_dimensionality_reduction(self, input, **kwargs):
        return input

    def apply_dimensionality_reduction(self, input, **kwargs):
        return input

    def cross_validation(self, X, Y, custom_params=None, **kwargs):
        # TODO with strategy: {self.cv_params['strategy']}")

        model_params = custom_params if custom_params is not None else self.model_params
        instantiate_model = lambda: self.instantiate_model(**model_params)
        cv_raw = cross_validate(instantiate_model, X, Y, **self.cv_params)
        cv_out = self.process_cv_out(cv_raw)
        return cv_out

    def process_cv_out(self, cv_raw):
        return pd.DataFrame(cv_raw)

    def fit_model(self, X, Y, **kwargs):
        self._trained_model = self.instantiate_model(**self.model_params).fit(X, Y, **kwargs)

    def pre_process_target_for_dim_reduction(self, Y):
        return Y

    def full_pipeline(self, refit=True, perform_cross_validation=True):
        X, Y = self.train_input, self.train_target

        target_supervised_embedding = self.pre_process_target_for_dim_reduction(Y)

        logger.debug(f"{self.model_label} - applying dimensionality reduction")
        X = self.fit_and_apply_dimensionality_reduction(
            input=X, Y=target_supervised_embedding, runtime_labelling=self.problem_label, read_cache=True
        )
        logger.debug(f"{self.model_label} - applying dimensionality reduction - Done")

        logger.info(f"{self.model_label} - performing cross validation")
        cv_out = self.cross_validation(X, Y)
        logger.info(
            f"{self.model_label} - Average  metrics: " + " | ".join([f"({k})={v:.4}" for k, v in cv_out.mean().items()])
        )

        if refit or not perform_cross_validation:
            self.fit_model(X, Y)
            Y_test = self.predict_public_test()
            self.generate_public_test_output(Y_test)

        return cv_out

    """
    def tuning(self):
        params = {"learning_rate": 0.1,
    "objective": "regression",
    "metric": "rmse",  # mae',
    "random_state": 0,
    "reg_alpha": 0.03,
    "reg_lambda": 0.002,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "max_depth": 10,
    "num_leaves": 186,
    "min_child_samples": 263}
        X, Y = self.train_input, self.train_target
        logger.debug(f"{self.model_label} - applying dimensionality reduction")
        X = self.fit_and_apply_dimensionality_reduction(input=X, runtime_labelling=self.problem_label)
        logger.debug(f"{self.model_label} - applying dimensionality reduction - Done")
        logger.info(f"{self.model_label} - performing cross validation")
        cv_out = self.cross_validation(X, Y, custom_params=self.build_model_params_for_tuning(params))
        logger.info(f"{self.model_label} - Average  metrics: " + " | ".join(
            [f"({k})={v:.4}" for k, v in cv_out.mean().items()]))
    """
    def optuna_pipeline(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optuna_objective, n_trials=3)
        print("Number of finished trials:", len(study.trials))
        print("Best trial:", study.best_trial.params)

        ...

    def optuna_objective(self, trial):
        params = {
            "metric": "rmse",
            "random_state": self.seed,
            "n_estimators": 20000,
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "subsample": trial.suggest_categorical("subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
            "max_depth": trial.suggest_categorical("max_depth", [10, 20, 100]),
            "num_leaves": trial.suggest_int("num_leaves", 1, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
            "cat_smooth": trial.suggest_int("min_data_per_groups", 1, 100),
        }

        X, Y = self.train_input, self.train_target
        logger.debug(f"{self.model_label} - applying dimensionality reduction")
        X = self.fit_and_apply_dimensionality_reduction(input=X, runtime_labelling=self.problem_label)
        logger.debug(f"{self.model_label} - applying dimensionality reduction - Done")
        logger.info(f"{self.model_label} - performing cross validation")
        cv_out = self.cross_validation(X, Y, custom_params=self.build_model_params_for_tuning(params))
        logger.info(
            f"{self.model_label} - Average  metrics: " + " | ".join([f"({k})={v:.4}" for k, v in cv_out.mean().items()])
        )

        vals = [v for k, v in cv_out.mean().items()]
        #todo mod
        return vals[0]

    def build_model_params_for_tuning(self, params):
        return params

    def predict_public_test(self) -> np.array:
        X_test_reduced = self.apply_dimensionality_reduction(
            input=self.test_input, runtime_labelling=f"{self.problem_label}_public_test", read_cache=True
        )
        return self._trained_model.predict(X_test_reduced)

    def generate_public_test_output(self, test_output: np.array):
        # test_output[:self.invalid_test_index] = 0
        submission = pd.read_csv(app_static_dir("data") / "sample_submission.csv", index_col="row_id").squeeze(
            "columns"
        )
        submission.iloc[self.public_test_index : len(test_output.ravel())] = test_output.ravel()
        assert not submission.isna().any()
        submission.to_csv(
            app_static_dir("out") / f"{self.model_label}_{datetime.now().strftime('%Y%m%d-%H%M')}_submission.csv"
        )
