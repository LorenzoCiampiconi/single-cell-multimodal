import lightgbm as lgb

from single_cell_multimodal_core.models.cite.base import CiteModelABC
from single_cell_multimodal_core.utils.log import setup_logging


class LGBMBasedModel(CiteModelABC):
    @property
    def model_class(self):
        return lgb.LGBMRegressor


if __name__ == "__main__":
    from single_cell_multimodal_core.models.cite.configurations.lgbm_first_try import configuration, model_label

    setup_logging("DEBUG")
    model_wrapper = LGBMBasedModel(configuration=configuration, label=model_label)

    model_wrapper.full_pipeline()
