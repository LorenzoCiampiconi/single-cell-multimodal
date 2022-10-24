import abc

import numpy as np


class DimensionalityReductionMixinABC(metaclass=abc.ABCMeta):
    configuration: dict

    @abc.abstractmethod
    def apply_dimensionality_reduction(self, *, input) -> np.array:
        pass

    @abc.abstractmethod
    def fit_and_apply_dimensionality_reduction(self, *, input, output_dimensionality=64) -> np.array:
        pass