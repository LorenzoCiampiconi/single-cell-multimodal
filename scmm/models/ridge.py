from sklearn.linear_model import Ridge


class RidgeMixin:
    @property
    def model_class(self):
        return Ridge

    def tune_model(self):
        pass
