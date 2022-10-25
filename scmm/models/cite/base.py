import abc
import logging

import numpy as np
from scipy import sparse

from scmm.utils.data_handling import load_sparse
from scmm.models.base import SCMModelABC

logger = logging.getLogger(__name__)


class CiteModelABC(SCMModelABC, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._train_input = None
        self._train_target = None
        self._test_input = None

    @property
    def problem_label(self) -> str:
        return "cite"

    @property
    def train_input(self) -> sparse.csr_array:
        if self._train_input is None:
            logger.info(f"{self.model_label} is loading training input from CITE dataset")
            self._train_input = load_sparse(split="train", problem="cite", type="inputs")
        return self._train_input

    @property
    def train_target(self) -> np.array:
        if self._train_target is None:
            logger.info(f"{self.model_label} is loading training target from CITE dataset")
            self._train_target = load_sparse(split="train", problem="cite", type="targets")
        return self._train_target.toarray()

    @property
    def test_input(self) -> sparse.csr_array:
        if self._test_input is None:
            logger.info(f"{self.model_label} is loading test input from CITE dataset")
            self._test_input = load_sparse(split="test", problem="cite", type="inputs")
        return self._test_input
