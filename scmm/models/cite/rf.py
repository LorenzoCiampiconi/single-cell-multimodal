# noinspection PyInterpreter
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressorMixin:
    @property
    def model_class(self):
        return RandomForestRegressor