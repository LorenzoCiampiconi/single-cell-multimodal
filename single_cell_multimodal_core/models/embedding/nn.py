import abc
import logging

import numpy as np
from torch import nn

logger = logging.getLogger()


class NNEntity(nn.Module):
    def __init__(self, *, activation_function=None):
        super().__init__()
        if activation_function is None:
            activation_function = nn.ReLU
        self._activation_function = activation_function

    @property
    def activation_function(self):
        return self._activation_function

    @property
    def depth(self) -> int:
        return np.NaN #todo

class FullyConnectedMixin(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _build_fallback_fully_connected(self):
        pass

    @abc.abstractmethod
    def validate_input_sequential(self, fully_connected_sequential):
        pass

    def sanitize_fully_connected_sequential(self, fully_connected_sequential):
        if fully_connected_sequential is not None and not self.validate_input_sequential(fully_connected_sequential):  # todo
            logger.debug(
                "loading passed by argument sequential is not verified, a fallback fully connected layer will be built."
            )
            fully_connected_sequential = self._build_fallback_fully_connected()
        else:
            logger.debug("loading passed by argument sequential is verified and loaded.")
        return fully_connected_sequential