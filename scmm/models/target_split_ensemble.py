import abc
import logging
from typing import List, Tuple, Dict, Type, Union

import numpy as np
import pandas as pd
from scipy import sparse

from scmm.models.base_model import SCMModelABC

logger = logging.getLogger(__name__)


class SCMModelTargetSubsetMixin:
    def __init__(self, *args, target_column_subset: List[int], **kwargs):
        super().__init__(*args, **kwargs)
        self._target_column_subset = target_column_subset

    @property
    def target_column_subset(self):
        return self._target_column_subset


class EnsembleTargetSubsetWSCMMModelEstimator(metaclass=abc.ABCMeta):
    def __init__(self, *, subset_mapping:dict, output_dim):
        self._subsets_mapping = subset_mapping
        self._output_dim = output_dim

        for subset, model_definition in self._subsets_mapping.items():
            assert "model_class" in model_definition, f"no model class specified for subset {subset}"
            assert (
                "configuration" in model_definition
            ), f"no configuration passed for model class {model_definition['model_class']} solving the subset {subset}"

        self._subset_models: Union[Dict[Tuple[int], type[SCMModelTargetSubsetMixin, SCMModelABC]], None] = None

    @staticmethod
    def _extend_model_w_target_subset_mixin(model_class: Type[SCMModelABC]):
        return type(f"{model_class}Subset", (SCMModelTargetSubsetMixin, model_class), {})

    @property
    def set_all_targets(self):
        all_targets = []
        for model in self._subset_models.values():
            all_targets += model.target_column_subset

        return set(all_targets)

    def _instantiate_subset_models(self) -> None:
        subset_models = dict()

        for subset in self._subsets_mapping:
            model_definition = self._subsets_mapping[subset]
            subset_model_class = self._extend_model_w_target_subset_mixin(model_definition["model_class"])
            subset_model = subset_model_class(configuration=model_definition["configuration"], target_column_subset=subset)
            # subset_model
            subset_models[subset] = subset_model

        self._subset_models = subset_models

        assert len(self.set_all_targets) == self._output_dim

    @property
    def is_instantiated(self):
        return self._subset_models is not None and all(model.is_instantiated for model in self._subset_models.values())

    @property
    def is_fit(self):
        return self.is_instantiated and all(model.is_fit for model in self._subset_models.values())

    def instantiate_estimator(self, **model_instantiation_kwargs):
        if len(model_instantiation_kwargs) > 0:
            logger.warning("argument to ensemble instantiation has been passed, they will be discarded")

        self._instantiate_subset_models()

    def fit(self, X, Y, **kwargs):
        if not self.is_instantiated:
            self._instantiate_subset_models()

        for estimator in self._subset_models.values():
            estimator: type[SCMModelTargetSubsetMixin, SCMModelABC]
            estimator.fit(X, Y[:, estimator.target_column_subset], **kwargs)

        return self

    def predict(self, X, **kwargs):
        assert self.is_fit

        output = np.zeros((X.shape[0], self._output_dim))

        for estimator in self._subset_models.values():
            estimator: type[SCMModelTargetSubsetMixin, SCMModelABC]
            output[:, estimator.target_column_subset] = estimator.predict(X, **kwargs)

        return output


class EnsembleTargetSubsetWSCMMModelEstimatorMixin:
    @property
    def estimator_class(self):
        return EnsembleTargetSubsetWSCMMModelEstimator

    def _is_estimator_fit(self, estimator):
        return estimator.is_fit
