import logging

import numpy as np
# import numpy as np
from sklearn.decomposition import TruncatedSVD

from single_cell_multimodal_core.models.dimensionality_reduction.base import DimensionalityReductionMixinABC

# from single_cell_multimodal_core.utils.caching import caching_function

logger = logging.getLogger(__name__)

class TruncatedSVDMixin(DimensionalityReductionMixinABC):
    seed: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._svd: TruncatedSVD = None


    # todo @caching_function(file_label="svd", file_extension="npz", loading_function=np.load, labelling_kwargs=, cache_folder=None)
    def fit_and_apply_dimensionality_reduction(self, *, input, output_dimensionality=64) -> np.array:

        logger.info(f"instatiating svd")
        self._svd = TruncatedSVD(n_components=output_dimensionality, random_state=self.seed)
        logger.info(f"fitting and applying svd to an input of dimension {input.shape[1]}, with target dimensionality of {output_dimensionality}")
        output = self._svd.fit_transform(input)
        logger.info(f"fitting and applying svd to an input of dimension {input.shape[1]}, with target dimensionality of {output_dimensionality} - Done")

        logger.debug(f"Reduced X shape to {output_dimensionality} dimension with SVD, the memory usage is now:"
                     f"  {str(input.shape):14} {input.size*4/1024/1024/1024:2.3f} GByte")

        return output

    def apply_dimensionality_reduction(self, *, input) -> np.array:
        logger.info(f"applying svd to an input of dimension {input.shape[1]}")
        output = self._svd.transform(input)
        logger.info(f"applying svd to an input of dimension {input.shape[1]} - Done")

        logger.debug(f"Reduced X shape to {self._svd.n_components} dimension with SVD, the memory usage is now:"
                     f"  {str(input.shape):14} {input.size*4/1024/1024/1024:2.3f} GByte")

        return output