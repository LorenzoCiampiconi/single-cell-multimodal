from single_cell_multimodal_core.models.cite.base import CiteModelABC
from single_cell_multimodal_core.models.cite.lgbm import LGBMMixin
from single_cell_multimodal_core.models.dimensionality_reduction.svd import TruncatedSVDMixin
from single_cell_multimodal_core.utils.log import setup_logging


class LGBMwSVDCite(LGBMMixin, TruncatedSVDMixin, CiteModelABC):
    ...


if __name__ == "__main__":
    from single_cell_multimodal_core.models.cite.configurations.lgbm_first_try import configuration, model_label

    setup_logging("DEBUG")
    model_wrapper = LGBMwSVDCite(configuration=configuration, label=model_label)

    model_wrapper.full_pipeline(save_model=True)