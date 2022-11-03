# noinspection PyInterpreter
import lightgbm as lgb


class LGBMMixin:
    @property
    def model_class(self):
        return lgb.LGBMRegressor

    def tune_model(self):
        pass
