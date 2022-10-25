import abc
import logging

import numpy as np
from torch import nn

from scmm.models.embedding.autoencoder.utils import init_fc_snn

logger = logging.getLogger()


class NNEntity(nn.Module):
    def __init__(self, *, activation_function=None):
        super().__init__()
        if activation_function is None:
            activation_function = nn.ReLU
        self._activation_function = activation_function

    def reset_parameters(self):
        self.apply(init_fc_snn)

    @property
    def activation_function(self):
        return self._activation_function

    @property
    def depth(self) -> int:
        return np.NaN  # todo


class FullyConnectedSequentialNotInstatiated(Exception):
    ...


class FullyConnectedMixin(metaclass=abc.ABCMeta):
    def __init__(self, fully_connected_sequential=None, build_fallback=True, **kwargs):
        super().__init__(**kwargs)
        self._build_fallback = build_fallback
        self._fc = fully_connected_sequential

    @abc.abstractmethod
    def _build_fallback_fully_connected(self):
        pass

    @abc.abstractmethod
    def validate_input_sequential(self, fully_connected_sequential):
        pass

    def forward(self, x):
        return self._fc(x)

    def sanitize_fc(self):
        if self._fc is not None:
            ok = self.validate_input_sequential(self._fc)
            if not ok:
                logger.warning("The instatiated sequential has not passed sanity checks")
                if self._build_fallback:
                    logger.warning("a fully connected fallback layer will be built.")
                    self._fc = self._build_fallback_fully_connected()
                else:
                    assert (
                        ok,
                        "The instatiated sequential is not consistent with the expected constraint of the module. Abort",
                    )
            else:
                logger.debug("Instatiated sequential is verified and loaded.")
        else:
            raise FullyConnectedSequentialNotInstatiated()
