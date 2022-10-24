import logging
from typing import List, Type

import numpy as np

from single_cell_multimodal_core.models.dimensionality_reduction.base import DimensionalityReductionMixinABC

logger = logging.getLogger(__name__)


class DReductorInstatiableMixin:
    def __init__(self, *, seed, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed


class MultiLevelDimensionalityReductionWrapperMixin(DimensionalityReductionMixinABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_dims: List[int] = self.configuration["dimensionality_reduction_params"][
            "intermediate_dimensionalities"
        ]
        self._dimensionality_reducers_class: List[Type[DimensionalityReductionMixinABC]] = [
            self._make_reductor_type_instatiable(reductor_type)
            for reductor_type in self.configuration["dimensionality_reduction_params"]["dimensionality_reducers_class"]
        ]
        self.dimensionality_reducers = None

        assert len(self.intermediate_dims) > 0
        assert len(self._dimensionality_reducers_class) == len(self.intermediate_dims) + 1

    def _make_reductor_type_instatiable(
        self, original_type: Type[DimensionalityReductionMixinABC]
    ) -> Type[DimensionalityReductionMixinABC]:
        """
        :rtype: Type[DimensionalityReductionMixinABC]
        """
        return type(f"{original_type}_instatiable", (DReductorInstatiableMixin, original_type), {})

    def _instantiate_reducers(self, force=False):
        if self.dimensionality_reducers is None or force:
            dimensionality_reducers = []
            for dimensionality_reducer_class in self._dimensionality_reducers_class:
                dimensionality_reducers.append(
                    dimensionality_reducer_class(seed=self.seed)
                )  # todo, this is not instatiating the seed, develop it
            self.dimensionality_reducers = dimensionality_reducers

    def apply_dimensionality_reduction(self, *, input) -> np.array:
        latent_dim = self.intermediate_dims[0]
        for i, reducer in enumerate(self.apply_dimensionality_reduction[:-1]):
            reducer.apply_dimensionality_reduction(input=input)
            input = latent_dim
            latent_dim = self.intermediate_dims[i + 1]

        return self.dimensionality_reducers[-1].apply_dimensionality_reduction(input=latent_dim)

    def fit_and_apply_dimensionality_reduction(self, *, input, output_dimensionality=64) -> np.array:
        self._instantiate_reducers()

        for i, reducer in enumerate(self.dimensionality_reducers[:-1]):
            latent_dim = self.intermediate_dims[i]
            logger.info(
                f"Applying dimensionality reduction step {i}, from {input.shape[1]} to {latent_dim}, with {reducer}"
            )
            input = reducer.fit_and_apply_dimensionality_reduction(input=input, output_dimensionality=latent_dim)

        return self.dimensionality_reducers[-1].fit_and_apply_dimensionality_reduction(
            input=input, output_dimensionality=output_dimensionality
        )
