from scmm.utils.metrics import correlation_score
from sklearn.metrics import make_scorer

common_metrics = {
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "explained_variance": "explained_variance",
    "correlation_score": make_scorer(correlation_score, greater_is_better=True),
}
