import numpy as np
import pandas as pd


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


# def negative_correlation_loss(y_true, y_pred):
#     """Negative correlation loss function for Keras
#
#     Precondition:
#     y_true.mean(axis=1) == 0
#     y_true.std(axis=1) == 1
#
#     Returns:
#     -1 = perfect positive correlation
#     1 = totally negative correlation
#     """
#     my = K.mean(tf.convert_to_tensor(y_pred), axis=1)
#     my = tf.tile(tf.expand_dims(my, axis=1), (1, y_true.shape[1]))
#     ym = y_pred - my
#     r_num = K.sum(tf.multiply(y_true, ym), axis=1)
#     r_den = tf.sqrt(K.sum(K.square(ym), axis=1) * float(y_true.shape[-1]))
#     r = tf.reduce_mean(r_num / r_den)
#     return -r
