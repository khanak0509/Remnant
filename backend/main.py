import asyncio
import json
import os

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.stats import norm

from config import FLAG_PROB, FLEET_SUBSET, HORIZON, MODEL_DIR, SETTINGS, SENSORS, SUBSETS, WINDOW, Z90
from data import add_test_rul, pad, read, regimes, replay_windows, scale

app = FastAPI(title="Remnant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

prep = {s: joblib.load(f"{MODEL_DIR}/prep_{s}.joblib") for s in SUBSETS}
sessions = {s: ort.InferenceSession(f"{MODEL_DIR}/remnant_{s}.onnx", providers=["CPUExecutionProvider"]) for s in SUBSETS}
delay = float(os.getenv("STREAM_DELAY", "0.3"))
status = {"subset": FLEET_SUBSET, "engines": 0, "flagged": 0, "last_cycle": 0}


class Window(BaseModel):
    subset: str = "FD001"
    window: list[list[float]]


def transform(df, subset):
    p = prep[subset]
    df["regime"] = regimes(p["kmeans"], df)
    return scale(df, p["scalers"], p["sensors"])


def run(subset, X):
    mean, sd = sessions[subset].run(None, {"window": X.astype(np.float32)})
    return np.clip(mean, 0, None), sd


def summary(mean, sd):
    p = norm.cdf((HORIZON - mean) / sd)
    watch = norm.cdf((2 * HORIZON - mean) / sd)
    flag = p > FLAG_PROB
    state = np.where(flag, "maintain", np.where(watch > FLAG_PROB, "watch", "ok"))
    return {"rul": mean, "lower": np.clip(mean - Z90 * sd, 0, None), "upper": mean + Z90 * sd,
            "p_fail": p, "flag": flag, "status": state}


def fleet():
    _, test, rul = read(FLEET_SUBSET)
    test = transform(add_test_rul(test, rul), FLEET_SUBSET)
    X, units, cycles, true = replay_windows(test, prep[FLEET_SUBSET]["sensors"])
    out = summary(*run(FLEET_SUBSET, X))
    out.update({"unit": units, "cycle": cycles, "true_rul": true})
    return out


replay = fleet()
status["engines"] = int(len(np.unique(replay["unit"])))


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(req: Window):
    x = pad(np.array(req.window, dtype=float))[-WINDOW:]
    df = transform(pd.DataFrame(x, columns=SETTINGS + SENSORS), req.subset)
    X = df[prep[req.subset]["sensors"]].values[None]
    r = summary(*run(req.subset, X))
    return {"subset": req.subset, "rul": round(float(r["rul"][0]), 2), "lower": round(float(r["lower"][0]), 2),
            "upper": round(float(r["upper"][0]), 2), "p_fail": round(float(r["p_fail"][0]), 4),
            "flag": bool(r["flag"][0]), "status": str(r["status"][0]), "horizon": HORIZON}


def event(i):
    return {"unit": int(replay["unit"][i]), "cycle": int(replay["cycle"][i]), "rul": round(float(replay["rul"][i]), 1),
            "lower": round(float(replay["lower"][i]), 1), "upper": round(float(replay["upper"][i]), 1),
            "p_fail": round(float(replay["p_fail"][i]), 3), "flag": bool(replay["flag"][i]),
            "status": str(replay["status"][i]), "true_rul": round(float(replay["true_rul"][i]), 1)}


@app.get("/fleet/stream")
async def stream(speed: float = None, max_cycles: int = None):
    wait = delay if speed is None else speed
    last = int(replay["cycle"].max()) if max_cycles is None else max_cycles

    async def gen():
        flagged = set()
        for c in range(1, last + 1):
            idx = np.where(replay["cycle"] == c)[0]
            engines = [event(i) for i in idx]
            for e in engines:
                (flagged.add if e["flag"] else flagged.discard)(e["unit"])
            status["last_cycle"], status["flagged"] = c, len(flagged)
            yield f"data: {json.dumps({'cycle': c, 'engines': engines})}\n\n"
            await asyncio.sleep(wait)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/fleet/status")
def fleet_status():
    return status
