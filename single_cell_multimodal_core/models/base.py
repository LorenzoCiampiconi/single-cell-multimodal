import abc
import gc
import logging
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
    @property
    @abc.abstractmethod
    def configuration(self):
        pass

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
    @abc.abstractmethod
    def params(self):
        pass

    def apply_SVD(self, input, n_components=64):
        svd = TruncatedSVD(n_components=n_components, random_state=1)
        output = svd.fit_transform(input)

        logger.debug(f"Reduced X shape:  {str(input.shape):14} {input.size*4/1024/1024/1024:2.3f} GByte")

        return output

    def cross_validation(self, X, Y, submit):
        kf = KFold(n_splits=self.configuration['n_split_for_kfold'], shuffle=True, random_state=1)
        score_list = []
        # va_pred = []
        for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
            model = None
            gc.collect()
            X_tr = X[idx_tr]
            y_tr = Y[idx_tr]

            model = self.model_wrapper_class(self.model_class(**self.params))
            model.fit(X_tr, y_tr)
            del X_tr, y_tr
            gc.collect()

            if submit:
                # model.save(f"/kaggle/temp/model_{fold}")
                model_path: pathlib.Path = app_static_dir("SAVED_MODELS") / f"model_{fold}.pkl"

                model_path.write_bytes(pkl.dumps(model))

            # We validate the model
            X_va = X[idx_va]
            y_va = Y[idx_va]

            y_va_pred = model.predict(X_va)
            # va_pred.append(y_va_pred)
            mse = mean_squared_error(y_va, y_va_pred)
            corrscore = correlation_score(y_va, y_va_pred)
            del X_va, y_va

            logger.info(f"Fold {fold}: mse = {mse:.5f}, corr =  {corrscore:.3f}")
            score_list.append((mse, corrscore))

        # Show overall score
        result_df = pd.DataFrame(score_list, columns=['mse', 'corrscore'])
        logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}Average  mse = {result_df.mse.mean():.5f}; corr = {result_df.corrscore.mean():.3f}{Style.RESET_ALL}")
