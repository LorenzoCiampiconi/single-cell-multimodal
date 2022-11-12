import abc
import logging

import pandas as pd
import numpy as np
from scipy import sparse
from scmm.utils.appdirs import app_static_dir

from scmm.utils.data_handling import load_sparse
from scmm.models.base_model import SCMModelABC

logger = logging.getLogger(__name__)


class MultiomeModelABC(SCMModelABC, metaclass=abc.ABCMeta):
    public_test_index: int = 6812820

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._train_input = None
        self._train_target = None
        self._test_input = None

    @property
    def problem_label(self) -> str:
        return "multiome"

    @property
    def train_input(self) -> sparse.csr_array:
        if self._train_input is None:
            logger.info(f"{self.model_label} is loading training input from MULTIOME dataset")
            self._train_input = load_sparse(split="train", problem="multi", type="inputs")
        return super().train_input

    @property
    def train_target(self) -> np.array:
        if self._train_target is None:
            logger.info(f"{self.model_label} is loading training target from MULTIOME dataset")
            self._train_target = load_sparse(split="train", problem="multi", type="targets")
        return super().train_target

    @property
    def test_input(self) -> sparse.csr_array:
        if self._test_input is None:
            logger.info(f"{self.model_label} is loading test input from MULTIOME dataset")
            self._test_input = load_sparse(split="test", problem="multi", type="inputs")
        return super().test_input

    @property
    def test_input_idx(self):
        if self._test_input_idx is None:
            # cols from target
            path = app_static_dir("data") / f"test_multi_inputs_idxcol.npz"
            with np.load(path, allow_pickle=True) as npz_file:
                index = npz_file["index"]
            # row from test
            path = app_static_dir("data") / f"train_multi_targets_idxcol.npz"
            with np.load(path, allow_pickle=True) as npz_file:
                columns = npz_file["columns"]

            self._test_input_idx = (index, columns)

        return self._test_input_idx

    def generate_submission_output(self, test_output: np.array):
        index, columns = self.test_input_idx
        df = pd.DataFrame(test_output, index=index, columns=columns)
        df.index.name = "cell_id"
        df.columns.name = "gene_id"
        eval_ids = pd.read_csv(app_static_dir("data") / "evaluation_ids.csv").set_index(["cell_type", "gene_id"])

        s = df.stack()
        s.name = "target"
        out = eval_ids.join(df, on=["cell_id", "gene_id"])
        out = out[["row_id", "target"]].set_index("target")

        assert not out.isna().any()

        return out
