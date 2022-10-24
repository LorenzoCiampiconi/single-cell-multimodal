import abc
import logging

import numpy as np

logger = logging.getLogger()

class Embedder(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def is_fit(self) ->bool:
        pass

    @abc.abstractmethod
    def fit(self, input):
        pass

    @abc.abstractmethod
    def transform(self, input) -> np.array:
        pass

    def fit_transform(self, input) -> np.array:
        logger.info('Embedder is being fit')
        self.fit(input)
        logger.info('Embedder has been fit - Done')

        logger.info('Now transforming the input')
        return self.transform(input)