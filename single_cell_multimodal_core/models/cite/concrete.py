from single_cell_multimodal_core.models.base import MultiOutputRegressorMixin
from single_cell_multimodal_core.models.cite.base import CiteModelABC
from single_cell_multimodal_core.models.cite.lgbm import LGBMMixin
from single_cell_multimodal_core.models.dimensionality_reduction.multilevel import \
    MultiLevelDimensionalityReductionWrapperMixin
from single_cell_multimodal_core.models.dimensionality_reduction.svd import TruncatedSVDMixin
from single_cell_multimodal_core.utils.log import setup_logging


class LGBMwSVDCite(
    LGBMMixin,
    TruncatedSVDMixin,
    MultiOutputRegressorMixin,
    CiteModelABC
):
    ...

class LGBMwSVDAutoEncoderCite(
    LGBMMixin,
    MultiLevelDimensionalityReductionWrapperMixin,
    MultiOutputRegressorMixin,
    CiteModelABC
):
    ...


if __name__ == "__main__":
    from single_cell_multimodal_core.models.cite.configurations.lgbm_w_autoencoder import configuration, model_label

    setup_logging("DEBUG")
    model_wrapper = LGBMwSVDAutoEncoderCite(configuration=configuration, label=model_label)

    model_wrapper.full_pipeline(save_model=True)