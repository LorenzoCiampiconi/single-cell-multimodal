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


class SCMModelEnsembleTargetSubset(SCMModelABC, metaclass=abc.ABCMeta):
    @property
    def estimator_class(self):
        logger.warning("you are trying to retrieve the class of estimator of an ensemble, which can be of multiple type, something is wrong!")
        return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            "subsets_mapping" in self.configuration
        ), "no subset mappings found in the passed configuration, this ensemble cannot be instantiated"

        self._subsets_mapping: dict = self.configuration["subsets_mapping"]

        for subset, model_definition in self._subsets_mapping.items():
            assert "model_class" in model_definition, f"no model class specified for subset {subset}"
            assert (
                "configuration" in model_definition
            ), f"no configuration passed for model class {model_definition['model_class']} solving the subset {subset}"

        self._subset_models: Union[Dict[Tuple[int], type[SCMModelTargetSubsetMixin, SCMModelABC]], None] = None

    @staticmethod
    def _extend_model_w_target_subset_mixin(model_class: Type[SCMModelABC]):
        return type(f"{model_class}Subset", (SCMModelTargetSubsetMixin, model_class), {})

    def _instantiate_subset_models(self) -> None:
        subset_models = dict()

        for subset in self._subsets_mapping:
            model_definition = self._subsets_mapping[subset]
            subset_model_class = self._extend_model_w_target_subset_mixin(model_definition["model_class"])
            subset_model = subset_model_class(configuration=model_definition["configuration"], target_column_subset=subset)
            # subset_model
            subset_models[subset] = subset_model

        self._subset_models = subset_models

        assert all(m.problem_label == self.problem_label for m in subset_models.values())

        all_targets = []
        for model in self._subset_models.values():
            all_targets += model.target_column_subset

        assert len(set(all_targets)) == self.train_target.shape[1]

    @property
    def is_instantiated(self):
        return self._subset_models is not None and all(model.is_instantiated for model in self._subset_models.values())

    def is_fit(self):
        return self.is_instantiated and all(model.is_fit for model in self._subset_models.values())

    def instantiate_estimator(self, **model_instantiation_kwargs):
        if len(model_instantiation_kwargs) > 0:
            logger.warning("argument to ensemble instantiation has been passed, they will be discarded")

        self._instantiate_subset_models()

    def fit_estimator(self, X, Y, **kwargs):
        self._instantiate_subset_models()

        for estimator in self._subset_models.values():
            estimator: type[SCMModelTargetSubsetMixin, SCMModelABC]
            estimator.fit(X, Y[:, estimator.target_column_subset], **kwargs)

    def predict(self, X, runtime_labelling=None, **kwargs):
        assert self.is_fit

        runtime_labelling = (
            f"{self.problem_label}_{runtime_labelling}" if runtime_labelling is not None else f"{self.problem_label}"
        )

        write_cache = read_cache = not self._cv_mod

        X_r = self.apply_dimensionality_reduction(
            input=X, runtime_labelling=runtime_labelling, read_cache=read_cache, write_cache=write_cache
        )

        output = np.zeros((X_r.shape[0], self.train_target.shape[1]))

        for estimator in self._subset_models.values():
            estimator: type[SCMModelTargetSubsetMixin, SCMModelABC]
            output[:, estimator.target_column_subset] = estimator.predict(X_r)
            # todo add merge by mean

        return output

    def save_public_test_output(self, output: pd.DataFrame):
        super().save_public_test_output(output)

    def _is_estimator_fit(self, estimator):
        return True
