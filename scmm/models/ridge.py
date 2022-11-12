from sklearn.linear_model import Ridge


class RidgeMixin:
    @property
    def estimator_class(self):
        return Ridge

    def tune_model(self):
        pass

    def _is_estimator_fit(self, estimator):
        return True  # todo
