import abc
from datetime import datetime
import gc
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from single_cell_multimodal_core.metrics import correlation_score
from single_cell_multimodal_core.utils.appdirs import app_static_dir

logger = logging.getLogger(__name__)


class SCMModelABC:
    invalid_test_index = 7476

    def __init__(self, configuration: Dict[str, Dict[str, Any]], label=""):
        self._configuration = configuration
        self._model_label = f"{self.__class__.__name__}-{label}" if label else self.__class__.__name__
        self._trained_model = None

    @property
    def model_label(self):
        return self._model_label

    @property
    def is_trained(self):
        return self._trained_model is not None

    @property
    @abc.abstractmethod
    def configuration(self):
        return self._configuration

    @property
    @abc.abstractmethod
    def train_input(self) -> scipy.sparse.csr.csr_matrix:
        pass

    @property
    @abc.abstractmethod
    def train_target(self) -> scipy.sparse.csr.csr_matrix:
        pass

    @property
    @abc.abstractmethod
    def test_input(self) -> scipy.sparse.csr.csr_matrix:
        pass

    @property
    def model_wrapper_class(self):
        return MultiOutputRegressor

    @property
    @abc.abstractmethod
    def model_class(self):
        pass

    @property
    def cross_validation_params(self):
        return self.configuration["cross_validation_params"]

    @property
    def model_params(self):
        return self.configuration["model_params"]

    @property
    def seed(self):
        return self.configuration['global_params']['seed']

    @property
    def svd_params(self):
        return self.configuration["svd_params"]

    def apply_SVD(self, input, n_components=64):
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        output = svd.fit_transform(input)

        logger.debug(f"Reduced X shape:  {str(input.shape):14} {input.size*4/1024/1024/1024:2.3f} GByte")

        return output

    def cross_validation(self, X, Y, save_model=False):
        logger.info(f"{self.model_label} - performing cross validation")

        kf = KFold(n_splits=self.cross_validation_params["n_splits_for_kfold"], shuffle=True, random_state=1)
        score_list = []
        # va_pred = []
        for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
            model = None
            gc.collect()
            X_tr = X[idx_tr]
            y_tr = Y[idx_tr]

            model = self.model_wrapper_class(self.model_class(**self.model_params))
            model.fit(X_tr, y_tr)
            del X_tr, y_tr
            gc.collect()

            if save_model:
                # model_wrapper.save(f"/kaggle/temp/model_{fold}")
                model_path: pathlib.Path = app_static_dir("SAVED_MODELS") / f"model_{self.model_label}_{fold}.pkl"

                model_path.write_bytes(pkl.dumps(model))

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

        # Show overall score
        result_df = pd.DataFrame(score_list, columns=["mse", "corrscore"])
        logger.info(
            f"{self.model_label} - Average  mse = {result_df.mse.mean():.5f}; corr = {result_df.corrscore.mean():.3f}"
        )

        self._trained_model = self.model_wrapper_class(self.model_class(**self.model_params)).fit(X, Y)

    def full_pipeline(self, save_model=False):
        X, Y = self.train_input, self.train_target

        logger.debug(f"{self.model_label} - applying SVD")
        X = self.apply_SVD(X, **self.svd_params)
        logger.debug(f"{self.model_label} - applying SVD - Done")

        self.cross_validation(X, Y, save_model=save_model)

        self.generate_public_test_output(self.apply_SVD(self.public_test(), **self.svd_params))

    def public_test(self) -> np.array:
        return self._trained_model.predict(self.test_input)

    def generate_public_test_output(self, test_output: np.array):
        test_output[:7476] = 0
        submission = pd.read_csv(app_static_dir("DATA") / "sample_submission.csv", index_col="row_id", squeeze=True)
        submission.iloc[: len(test_output.ravel())] = test_output.ravel()
        assert not submission.isna().any()
        submission.to_csv(app_static_dir('OUT') / f"{self.model_label}_{datetime.now().strftime('%Y%m%d-%H%M')}_submission.csv")
