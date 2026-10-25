import json

import numpy as np
from fastapi.testclient import TestClient

from config import COLUMNS
from data import read
from main import app

client = TestClient(app)


def test_predict_schema():
    test = read("FD001")[1]
    window = test[test["unit"] == 1][COLUMNS[2:]].values[-30:].tolist()
    r = client.post("/predict", json={"subset": "FD001", "window": window})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"rul", "lower", "upper", "p_fail", "flag", "status"}
    assert body["lower"] <= body["rul"] <= body["upper"]
    assert 0 <= body["p_fail"] <= 1


def test_predict_short_window():
    r = client.post("/predict", json={"subset": "FD002", "window": np.zeros((5, 24)).tolist()})
    assert r.status_code == 200


def test_fleet_stream_and_status():
    with client.stream("GET", "/fleet/stream?speed=0&max_cycles=3") as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {\"cycle")]
    assert [e["cycle"] for e in events] == [1, 2, 3]
    assert set(events[0]["engines"][0]) >= {"unit", "rul", "lower", "upper", "flag", "true_rul"}
    s = client.get("/fleet/status").json()
    assert s["engines"] == 100 and s["last_cycle"] == 3 and 0 <= s["flagged"] <= 100
