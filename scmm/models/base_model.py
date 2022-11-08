import abc
import logging
from datetime import datetime
from typing import Any, Dict

import optuna
import numpy as np
import pandas as pd
from scipy import sparse
from scmm.utils.appdirs import app_static_dir
from scmm.utils.scikit_crossval import cross_validate

logger = logging.getLogger(__name__)


class SCMModelABC(metaclass=abc.ABCMeta):
    invalid_test_index: int = 7476
    public_test_index: int

    def __init__(self, configuration: Dict[str, Dict[str, Any]], label="", cv_mod=False):
        self._configuration = configuration
        self._model_label = f"{label}" if label else self.__class__.__name__
        self._estimator = None
        self._cv_mod = cv_mod

    @staticmethod
    def now_string() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M")

    @property
    def cloning_params(self):
        return dict(configuration=self.configuration, label=self._model_label)

    @property
    def model_label(self):
        return f"{self.problem_label}_{self._model_label}"

    @property
    def is_trained(self):
        return self._estimator is not None

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

    @property
    def test_input_idx(self):
        pass

    def instantiate_estimator(self, **model_instantiation_kwargs):
        return self.model_class(**model_instantiation_kwargs)

    @property
    @abc.abstractmethod
    def model_class(self):
        pass

    @property
    def model_instantiation_kwargs(self):
        return self.model_params

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

    @property
    def is_fit(self):
        return self._estimator is not None and self._is_estimator_fit(self._estimator)

    @abc.abstractmethod
    def _is_estimator_fit(self, estimator):
        pass

    def fit_and_apply_dimensionality_reduction(self, input, **kwargs):
        return input

    def apply_dimensionality_reduction(self, input, **kwargs):
        return input

    def full_cross_validation(self, custom_params=None):
        model_params = custom_params if custom_params is not None else self.cloning_params

        instantiate_model = lambda: self.__class__(cv_mod=True, **model_params)
        X, Y = self.train_input, self.train_target

        cv_raw = cross_validate(instantiate_model, X, Y, **self.cv_params)
        cv_out = self.process_cv_out(cv_raw)
        return cv_out

    def set_params(self):
        pass  # todo

    def cross_validation_of_estimator(self, X, Y, custom_params=None, **kwargs):
        # TODO with strategy: {self.cv_params['strategy']}")
        model_params = custom_params if custom_params is not None else self.model_instantiation_kwargs
        instantiate_model = lambda: self.instantiate_estimator(**model_params)
        cv_raw = cross_validate(instantiate_model, X, Y, **self.cv_params)
        cv_out = self.process_cv_out(cv_raw)
        return cv_out

    def process_cv_out(self, cv_raw):
        return pd.DataFrame(cv_raw)

    def fit_estimator(self, X, Y, **kwargs):
        self._estimator = self.instantiate_estimator(**self.model_instantiation_kwargs).fit(X, Y, **kwargs)

    def pre_process_target_for_dim_reduction(self, Y):
        return Y

    def fit(self, X, Y, fold=None):
        target_supervised_embedding = self.pre_process_target_for_dim_reduction(Y)

        folding_message = f" - to fold {fold}" if fold is not None else ""
        logger.debug(f"{self.model_label} - applying dimensionality reduction{folding_message}")

        runtime_labelling = f"{self.problem_label}_{fold}fold" if fold is not None else self.problem_label

        write_cache = read_cache = save_checkpoints = not self._cv_mod

        X = self.fit_and_apply_dimensionality_reduction(
            input=X,
            Y=target_supervised_embedding,
            runtime_labelling=runtime_labelling,
            read_cache=read_cache,
            write_cache=write_cache,
            save_checkpoints=save_checkpoints,
        )
        logger.debug(f"{self.model_label} - applying dimensionality reduction {folding_message} - Done")

        logger.info(f"{self.model_label} - fitting estimator{folding_message}")
        self.fit_estimator(X, Y)
        logger.info(f"{self.model_label} - fitting estimator{folding_message} - Done")

    def predict(self, X, runtime_labelling=None, **kwargs):
        assert self.is_fit

        runtime_labelling = (
            f"{self.problem_label}_{runtime_labelling}" if runtime_labelling is not None else f"{self.problem_label}"
        )

        write_cache = read_cache = not self._cv_mod

        X_r = self.apply_dimensionality_reduction(
            input=X, runtime_labelling=runtime_labelling, read_cache=read_cache, write_cache=write_cache
        )
        return self._estimator.predict(X_r)

    def fit_and_predict_problem(self):
        self.fit(self.train_input, self.train_target)
        Y_hat = self.predict(self.test_input)
        np.savez(app_static_dir(f"out/{self.problem_label}") / f"{self.model_label}_{self.now_string()}.npz", Y_hat)

    def pipeline_w_fixed_embedding(self, refit=True, perform_cross_validation=True):
        X, Y = self.train_input, self.train_target

        target_supervised_embedding = self.pre_process_target_for_dim_reduction(Y)

        logger.debug(f"{self.model_label} - applying dimensionality reduction")
        X = self.fit_and_apply_dimensionality_reduction(
            input=X, Y=target_supervised_embedding, runtime_labelling=self.problem_label, read_cache=True
        )
        logger.debug(f"{self.model_label} - applying dimensionality reduction - Done")

        logger.info(f"{self.model_label} - performing cross validation")
        cv_out = self.cross_validation_of_estimator(X, Y)
        logger.info(
            f"{self.model_label} - Average  metrics: " + " | ".join([f"({k})={v:.4}" for k, v in cv_out.mean().items()])
        )

        if refit or not perform_cross_validation:
            self.fit_estimator(X, Y)
            Y_test = self.predict(self.test_input, runtime_labelling=f"public_test")
            out = self.generate_submission_output(Y_test)
            self.save_public_test_output(out)

        return cv_out

    def optuna_pipeline(self):
        study = optuna.create_study(direction="maximize")
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
        cv_out = self.cross_validation_of_estimator(X, Y, custom_params=self.build_model_params_for_tuning(params))
        logger.info(
            f"{self.model_label} - Average  metrics: " + " | ".join([f"({k})={v:.4}" for k, v in cv_out.mean().items()])
        )

        vals = [v for k, v in cv_out.mean().items()]

        return vals[0]

    def build_model_params_for_tuning(self, params):
        return params

    def predict_public_test(self) -> np.array:
        X_test_reduced = self.apply_dimensionality_reduction(
            input=self.test_input, runtime_labelling=f"{self.problem_label}_public_test", read_cache=True
        )
        return self._trained_model.predict(X_test_reduced)

    @abc.abstractmethod
    def generate_submission_output(self, test_output: np.array):
        pass

    def save_public_test_output(self, output: pd.DataFrame):
        output.to_csv(
            app_static_dir("out") / f"{self.model_label}_{datetime.now().strftime('%Y%m%d-%H%M')}_submission.csv"
        )