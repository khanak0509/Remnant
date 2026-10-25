import json
import os
import sys

import numpy as np
import torch
from torch import nn

from config import MODEL_DIR, N_MEMBERS, RUL_CAP, SEED, SUBSETS, Z90
from data import prepare
from metrics import nasa_score, rmse
from train import device, fit, load, predict


def best_arch():
    scores = {}
    for name in ["cnn", "lstm", "transformer"]:
        r = json.load(open(f"{MODEL_DIR}/results_{name}.json"))
        scores[name] = np.mean([r[s]["val_rmse"] for s in SUBSETS])
    return min(scores, key=scores.get), scores


def nll(p, y):
    return nn.functional.gaussian_nll_loss(p[:, 0], y, nn.functional.softplus(p[:, 1]) + 1e-4)


def member_path(arch, subset, i):
    return f"{MODEL_DIR}/ens_{arch}_{subset}_{i}.pt"


def load_ensemble(arch, subset, n_features):
    return [load(arch, subset, n_features, out=2, path=member_path(arch, subset, i)) for i in range(N_MEMBERS)]


def combine(outs):
    mu = outs[..., 0] * RUL_CAP
    var = (np.log1p(np.exp(outs[..., 1])) + 1e-4) * RUL_CAP ** 2
    mean = mu.mean(axis=0)
    sd = np.sqrt((var + mu ** 2).mean(axis=0) - mean ** 2)
    return mean, sd


def predict_dist(members, X):
    outs = np.stack([predict(m, X, torch.device("cpu")) for m in members])
    return combine(outs)


def interval(mean, sd):
    return np.clip(mean - Z90 * sd, 0, None), mean + Z90 * sd


def main(subsets):
    arch, scores = best_arch()
    print("val rmse by arch", {k: round(v, 2) for k, v in scores.items()}, "-> best", arch)
    path = f"{MODEL_DIR}/results_ensemble.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    for s in subsets:
        d = prepare(s)
        for i in range(N_MEMBERS):
            model, val, epochs = fit(arch, d, seed=SEED + i, out=2, loss_fn=nll, dev=device(arch))
            torch.save(model.state_dict(), member_path(arch, s, i))
            print(f"  {s} member {i} val rmse {val:.2f} epochs {epochs}", flush=True)
        members = load_ensemble(arch, s, len(d["sensors"]))
        mean, sd = predict_dist(members, d["Xte"])
        mean = np.clip(mean, 0, None)
        lo, hi = interval(mean, sd)
        vmean, _ = predict_dist(members, d["Xval"])
        results[s] = {"arch": arch, "val_rmse": rmse(d["yval"], vmean), "rmse": rmse(d["yte"], mean),
                      "rmse_raw": rmse(d["yte_raw"], mean), "nasa": nasa_score(d["yte"], mean),
                      "coverage": float(np.mean((d["yte"] >= lo) & (d["yte"] <= hi))),
                      "width": float(np.mean(hi - lo)), "n": int(len(mean))}
        r = results[s]
        print(f"{s} ensemble  test rmse {r['rmse']:.2f}  nasa {r['nasa']:.0f}  90% coverage {r['coverage']:.3f}  "
              f"mean width {r['width']:.1f}")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or SUBSETS)
