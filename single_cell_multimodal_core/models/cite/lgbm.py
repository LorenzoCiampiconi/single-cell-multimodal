import lightgbm as lgb

from single_cell_multimodal_core.models.cite.base import CiteModelABC


class LGBMBasedModel(CiteModelABC):
    @property
    def model_class(self):
        return lgb.LGBMRegressor


if __name__ == '__main__':
    from single_cell_multimodal_core.models.cite.configurations.lgbm_first_try import params, model_label, cross_validation_params
    model_wrapper = LGBMBasedModel(configuration={'model_params': params, "cross_validation_params": cross_validation_params}, label=model_label)

    model_wrapper.cross_validation(model_wrapper.train_input, model_wrapper.train_target)

