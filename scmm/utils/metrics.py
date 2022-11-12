import numpy as np
import pandas as pd


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient"""
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.squeeze(axis=1).to_numpy()
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.squeeze(axis=1).to_numpy()

    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0] # rowvar = False?
    return corrsum / len(y_true)
