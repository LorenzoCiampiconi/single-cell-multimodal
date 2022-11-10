import abc
from typing import List

from scmm.models.base_model import SCMModelABC


class SCMModelSubsetABC(SCMModelABC, metaclass=abc.ABCMeta):
    def __init__(self, *, target_column_subset: List[int], **kwargs):
        super().__init__(**kwargs)
        self._target_column_subset = target_column_subset
