import logging

from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class TruncatedSVDMixin:
    seed: int

    def apply_dimensionality_reduction(self, input, output_dimensionality=64):
        svd = TruncatedSVD(n_components=output_dimensionality, random_state=self.seed)
        output = svd.fit_transform(input)

        logger.debug(f"Reduced X shape:  {str(input.shape):14} {input.size*4/1024/1024/1024:2.3f} GByte")

        return output