import json
import time

import joblib
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from config import MODEL_DIR, RUL_CAP, SUBSETS, WINDOW
from data import prepare
from uncertainty import best_arch, load_ensemble, predict_dist

RUNS = 2000


class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        outs = torch.stack([m(x) for m in self.members])
        mu = outs[..., 0] * RUL_CAP
        var = (nn.functional.softplus(outs[..., 1]) + 1e-4) * RUL_CAP ** 2
        mean = mu.mean(0)
        sd = torch.sqrt((var + mu ** 2).mean(0) - mean ** 2)
        return mean, sd


def export(subset, arch):
    d = prepare(subset)
    members = load_ensemble(arch, subset, len(d["sensors"]))
    model = Ensemble(members).eval()
    joblib.dump({"sensors": d["sensors"], "kmeans": d["kmeans"], "scalers": d["scalers"]}, f"{MODEL_DIR}/prep_{subset}.joblib")
    path = f"{MODEL_DIR}/remnant_{subset}.onnx"
    x = torch.from_numpy(d["Xte"][:1])
    torch.onnx.export(model, (x,), path, input_names=["window"], output_names=["mean", "sd"],
                      dynamic_axes={"window": {0: "batch"}, "mean": {0: "batch"}, "sd": {0: "batch"}},
                      dynamo=False, opset_version=17)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    mean, sd = sess.run(None, {"window": d["Xte"]})
    tmean, tsd = predict_dist(members, d["Xte"])
    diff = float(max(np.abs(mean - tmean).max(), np.abs(sd - tsd).max()))
    return sess, d, diff


def bench(sess, X):
    times = []
    for i in range(RUNS + 100):
        x = X[i % len(X)][None]
        start = time.perf_counter()
        sess.run(None, {"window": x})
        if i >= 100:
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    arch, _ = best_arch()
    results = {"arch": arch, "window": WINDOW, "runs": RUNS}
    for s in SUBSETS:
        sess, d, diff = export(s, arch)
        print(f"{s} exported, max abs diff vs torch {diff:.5f}")
        results[s] = {"max_diff": diff}
    sess = ort.InferenceSession(f"{MODEL_DIR}/remnant_FD001.onnx", providers=["CPUExecutionProvider"])
    med, p95 = bench(sess, prepare("FD001")["Xte"])
    results["latency_median_ms"], results["latency_p95_ms"] = med, p95
    print(f"onnx cpu latency per engine window: median {med:.3f} ms  p95 {p95:.3f} ms over {RUNS} runs")
    with open(f"{MODEL_DIR}/results_onnx.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
