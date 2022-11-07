import gc

import numpy as np
from tqdm import tqdm

from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _fit_and_score,
    _insert_error_scores,
    _normalize_score_results,
)
from sklearn.utils import indexable


def cross_validate(
    instantiate_model,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    verbose=0,
    progress_bar=False,
    fit_params=None,
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y)

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(instantiate_model(), scoring)
    else:
        scorers = _check_multimetric_scoring(instantiate_model(), scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    results = []
    for train, test in tqdm(cv.split(X, y, groups), total=cv.get_n_splits(), desc="k-fold", disable=not progress_bar):
        res = _fit_and_score(
            instantiate_model(),
            X,
            y,
            scorers,
            train,
            test,
            verbose,
            None,
            fit_params,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
        )
        results.append(res)
        gc.collect()

    # For callable scoring, the return type is only known after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    return ret
