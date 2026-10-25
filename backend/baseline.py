import json
import os

import lightgbm as lgb
import numpy as np

from config import MODEL_DIR, SEED, SUBSETS, WINDOW
from data import prepare
from metrics import nasa_score, rmse

t = np.arange(WINDOW) - (WINDOW - 1) / 2


def features(X):
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    slope = np.einsum("t,nts->ns", t, X - mean[:, None, :]) / (t ** 2).sum()
    return np.hstack([last, mean, slope])


params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.03, "num_leaves": 31,
          "min_data_in_leaf": 50, "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
          "seed": SEED, "verbose": -1}


def train(d):
    dtr = lgb.Dataset(features(d["Xtr"]), d["ytr"])
    dval = lgb.Dataset(features(d["Xval"]), d["yval"])
    return lgb.train(params, dtr, 2000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {}
    for s in SUBSETS:
        d = prepare(s)
        model = train(d)
        model.save_model(f"{MODEL_DIR}/lgbm_{s}.txt", num_iteration=model.best_iteration)
        val_preds = model.predict(features(d["Xval"]), num_iteration=model.best_iteration)
        preds = np.clip(model.predict(features(d["Xte"]), num_iteration=model.best_iteration), 0, None)
        results[s] = {"val_rmse": rmse(d["yval"], val_preds), "rmse": rmse(d["yte"], preds),
                      "rmse_raw": rmse(d["yte_raw"], preds), "nasa": nasa_score(d["yte"], preds),
                      "iters": model.best_iteration}
        r = results[s]
        print(f"{s} lgbm  val rmse {r['val_rmse']:.2f}  test rmse {r['rmse']:.2f} (uncapped {r['rmse_raw']:.2f})  nasa {r['nasa']:.0f}  iters {r['iters']}")
    with open(f"{MODEL_DIR}/results_lgbm.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
