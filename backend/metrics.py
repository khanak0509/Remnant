import numpy as np


def rmse(y, preds):
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(y)) ** 2)))


def nasa_score(y, preds):
    d = np.asarray(preds, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
