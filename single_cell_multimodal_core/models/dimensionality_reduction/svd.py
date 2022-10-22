import logging

# import numpy as np
from sklearn.decomposition import TruncatedSVD

# from single_cell_multimodal_core.utils.caching import caching_function

logger = logging.getLogger(__name__)

class TruncatedSVDMixin:
    seed: int

    # todo @caching_function(file_label="svd", file_extension="npz", loading_function=np.load, labelling_kwargs=, cache_folder=None)
    def apply_dimensionality_reduction(self, *, input, output_dimensionality=64):
        svd = TruncatedSVD(n_components=output_dimensionality, random_state=self.seed)
        output = svd.fit_transform(input)

        logger.debug(f"Reduced X shape to {output_dimensionality} dimension with SVD, the memory usage is now:"
                     f"  {str(input.shape):14} {input.size*4/1024/1024/1024:2.3f} GByte")

        return output