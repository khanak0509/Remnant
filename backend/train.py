import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODEL_DIR, RUL_CAP, SEED, SUBSETS
from data import prepare
from metrics import nasa_score, rmse
from models import MODELS

EPOCHS = 60
PATIENCE = 10
BATCH = 256
LR = {"cnn": 3e-4, "lstm": 1e-3, "transformer": 3e-4}


def device(name=None):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and name != "lstm":
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, X, dev, batch=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(dev)).cpu())
    return torch.cat(out).numpy()


def fit(name, d, seed=SEED, out=1, loss_fn=None, dev=None):
    dev = dev or device(name)
    seed_all(seed)
    model = MODELS[name](d["Xtr"].shape[2], out=out).to(dev)
    loss_fn = loss_fn or (lambda p, y: nn.functional.mse_loss(p[:, 0], y))
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(d["Xtr"]), torch.from_numpy(d["ytr"] / RUL_CAP)),
                        batch_size=BATCH, shuffle=True, generator=g)
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = rmse(d["yval"], predict(model, d["Xval"], dev)[:, 0] * RUL_CAP)
        sched.step(val)
        if val < best:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, epoch + 1


def load(name, subset, n_features, out=1, path=None):
    model = MODELS[name](n_features, out=out)
    model.load_state_dict(torch.load(path or f"{MODEL_DIR}/{name}_{subset}.pt", map_location="cpu"))
    return model.eval()


def main(name, subsets):
    os.makedirs(MODEL_DIR, exist_ok=True)
    dev = device(name)
    path = f"{MODEL_DIR}/results_{name}.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    for s in subsets:
        d = prepare(s)
        start = time.time()
        model, val, epochs = fit(name, d, dev=dev)
        torch.save(model.state_dict(), f"{MODEL_DIR}/{name}_{s}.pt")
        preds = np.clip(predict(model, d["Xte"], dev)[:, 0] * RUL_CAP, 0, None)
        results[s] = {"val_rmse": val, "rmse": rmse(d["yte"], preds), "rmse_raw": rmse(d["yte_raw"], preds),
                      "nasa": nasa_score(d["yte"], preds), "epochs": epochs, "seconds": time.time() - start}
        r = results[s]
        print(f"{s} {name}  val rmse {r['val_rmse']:.2f}  test rmse {r['rmse']:.2f} (uncapped {r['rmse_raw']:.2f})  "
              f"nasa {r['nasa']:.0f}  epochs {epochs}  {r['seconds']:.0f}s on {dev}")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or SUBSETS)
