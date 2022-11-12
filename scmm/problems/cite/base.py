import abc
import logging

import pandas as pd
import numpy as np
from scipy import sparse

from scmm.models.target_split_ensemble import SCMModelEnsembleTargetSubset
from scmm.utils.appdirs import app_static_dir

from scmm.utils.data_handling import load_sparse
from scmm.models.base_model import SCMModelABC

logger = logging.getLogger(__name__)


class CiteModelMixin:
    public_test_index: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def problem_label(self) -> str:
        return "cite"

    @property
    def train_input(self) -> sparse.csr_array:
        if self._train_input is None:
            logger.info(f"{self.model_label} is loading training input from CITE dataset")
            self._train_input = load_sparse(split="train", problem="cite", type="inputs")
        return super().train_input

    @property
    def train_target(self) -> np.array:
        if self._train_target is None:
            logger.info(f"{self.model_label} is loading training target from CITE dataset")
            self._train_target = load_sparse(split="train", problem="cite", type="targets")
        return super().train_target

    @property
    def test_input(self) -> sparse.csr_array:
        if self._test_input is None:
            logger.info(f"{self.model_label} is loading test input from CITE dataset")
            self._test_input = load_sparse(split="test", problem="cite", type="inputs")
        return super().test_input

    @property
    def test_input_idx(self):
        if self._test_input_idx is None:
            # cols from target
            path = app_static_dir("data") / f"test_cite_inputs_idxcol.npz"
            with np.load(path, allow_pickle=True) as npz_file:
                index = npz_file["index"]
            # row from test
            path = app_static_dir("data") / f"train_cite_targets_idxcol.npz"
            with np.load(path, allow_pickle=True) as npz_file:
                columns = npz_file["columns"]

            self._test_input_idx = (index, columns)

        return self._test_input_idx

    def generate_submission_output(self, test_output: np.array):
        # index, cols = self.test_input_idx
        # df = pd.DataFrame(test_output, index=index, columns=cols)
        # df.index.name = "cell_type"
        # # df.columns.name = "gene_id"
        # eval_ids = pd.read_csv(app_static_dir("data") / "evaluation_ids.csv", index_col="row_id")

        out = pd.Series(test_output.ravel(), name="target").to_frame()
        out["target"].iloc[self.public_test_index : self.invalid_test_index] = 0
        out.index.name = "row_id"
        out = out.squeeze()

        return out


class CiteModelABC(CiteModelMixin,SCMModelABC, metaclass=abc.ABCMeta):
    pass


class CiteModelEnsembleTargetSubsetABC(CiteModelMixin, SCMModelEnsembleTargetSubset, metaclass=abc.ABCMeta):
    pass
