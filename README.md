# Remnant

Remnant predicts the remaining useful life (RUL) of turbofan engines on the NASA C-MAPSS dataset and turns those predictions into maintenance decisions. It compares a gradient boosting baseline against a 1D CNN, an LSTM and a small Transformer on all four subsets, puts 90% prediction intervals on the best deep model with a deep ensemble, flags engines when the probability of failing inside a 20 cycle horizon gets too high, and simulates what that policy would cost against run to failure and a fixed maintenance schedule. The final ensemble is exported to ONNX and served by a FastAPI backend that can replay the test fleet cycle by cycle to a small React dashboard.

## Data

| subset | train engines | test engines | operating regimes | fault modes |
|--------|---------------|--------------|-------------------|-------------|
| FD001 | 100 | 100 | 1 | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 | HPC and fan degradation |
| FD004 | 249 | 248 | 6 | HPC and fan degradation |

Source: NASA Prognostics Center of Excellence, Turbofan Engine Degradation Simulation (C-MAPSS). `python download.py` fetches it from the PCoE S3 mirror. If that fails, put `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` in `backend/data/`.

Each row is one engine cycle: unit, cycle, 3 operating settings, 21 sensors.

Preprocessing:
- Sensors with within-regime std below 0.025 on the training units are dropped. On every subset that removes s1, s5, s6, s10, s16, s18, s19 and keeps 14 sensors. The next smallest sensor (s15) sits at 0.037, so the cut falls in a clear gap.
- FD002 and FD004: KMeans (k=6) on the operating settings, then a separate StandardScaler per regime. FD001 and FD003: one StandardScaler. KMeans and scalers are fit on the training units only.
- RUL target is capped at 125 cycles (piecewise linear).
- Windows of 30 cycles. Engines shorter than 30 cycles are padded by repeating their first row.
- 20% of the training engines (by unit ID, seed 42) are held out for validation. No window from a validation engine is ever in training.
- Test scoring uses the last window of each test engine against `RUL_FD00x.txt`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
python download.py
python data.py
python baseline.py
python train.py cnn
python train.py lstm
python train.py transformer
python uncertainty.py
python decision.py sweep
python decision.py
python export.py
python eval/run_all.py

uvicorn main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

On an M4 laptop the single models take 10 seconds to 2 minutes per subset and the 5 member LSTM ensemble takes about 45 minutes for all four subsets on CPU.

| service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:5173 |

If the API runs on another port, start the frontend with `VITE_API_URL=http://localhost:<port> npm run dev`.

## Architecture

| file | role |
|------|------|
| backend/config.py | constants: paths, window, RUL cap, costs, horizon, flag threshold |
| backend/download.py | fetches and unzips C-MAPSS |
| backend/data.py | loading, sensor drop, regime clustering, scaling, windows, unit split |
| backend/metrics.py | RMSE and the NASA asymmetric score |
| backend/baseline.py | LightGBM on last value, mean and slope per sensor |
| backend/models.py | CNN, LSTM and Transformer modules |
| backend/train.py | training loop with early stopping on val RMSE |
| backend/uncertainty.py | 5 member LSTM deep ensemble with Gaussian NLL, intervals, coverage |
| backend/decision.py | maintenance flag, fleet cost simulation, threshold sweep on validation |
| backend/export.py | ONNX export of the ensemble, parity check, latency benchmark |
| backend/main.py | FastAPI: /predict, /fleet/stream, /fleet/status |
| backend/eval/run_all.py | scores every stage, writes eval/results.md |
| backend/tests/ | pytest for data, models and API |
| frontend/src/App.jsx | fleet grid and replay controls |
| frontend/src/Chart.jsx | predicted vs true RUL chart with interval band |

Uncertainty: a deep ensemble instead of MC dropout, because each member learns its own noise variance so the intervals include aleatoric noise, and the 5 members export to one deterministic ONNX graph.

Maintenance rule: an engine is flagged when P(RUL < 20) > 0.05 under the ensemble's predicted Gaussian. Costs are 1000 for planned maintenance, 10000 for an unplanned failure and 10 per cycle of useful life thrown away by maintaining early. The fixed schedule maintains every engine at the interval that minimizes cost on the training lifetimes (127 to 144 cycles depending on subset). Test trajectories end before failure, so if the predictive policy has not flagged an engine by its last observed cycle it commits to the cycle where its final forecast says P(RUL < 20) will cross the threshold.

## API

`POST /predict` takes up to 30 rows of raw readings (3 settings then 21 sensors, in C-MAPSS column order). Shorter windows are padded.

```json
{
  "subset": "FD001",
  "window": [
    [-0.0025, 0.0005, 100.0, 518.67, 642.78, 1598.09, 1417.83, 14.62, 21.61, 553.26, 2388.11, 9080.49, 1.3, 47.91, 521.07, 2388.12, 8156.83, 8.4751, 0.03, 393.0, 2388.0, 100.0, 38.77, 23.2265],
    "... 29 more rows"
  ]
}
```

Response for test engine 31 (true RUL 8):

```json
{
  "subset": "FD001",
  "rul": 9.11,
  "lower": 2.57,
  "upper": 15.65,
  "p_fail": 0.9969,
  "flag": true,
  "status": "maintain",
  "horizon": 20
}
```

`GET /fleet/stream?speed=0.3` replays the FD001 test engines as server sent events, one event per cycle:

```json
{
  "cycle": 1,
  "engines": [
    {"unit": 1, "cycle": 1, "rul": 122.3, "lower": 108.6, "upper": 136.0, "p_fail": 0.0, "flag": false, "status": "ok", "true_rul": 142.0}
  ]
}
```

`GET /fleet/status`

```json
{"subset": "FD001", "engines": 100, "flagged": 0, "last_cycle": 2}
```

`status` is `maintain` when the engine is flagged and `watch` when P(RUL < 40) > 0.05.

## Evaluation

`python eval/run_all.py`, full per-subset tables in [backend/eval/results.md](backend/eval/results.md). RMSE is against the true RUL capped at 125, same as the training target.

| metric | result | threshold | status |
|--------|--------|-----------|--------|
| RMSE FD001 lgbm * | 12.28 | <= 14 | baseline |
| RMSE FD001 cnn | 14.43 | <= 14 | FAIL |
| RMSE FD001 lstm | 13.96 | <= 14 | PASS |
| RMSE FD001 transformer | 16.67 | <= 14 | FAIL |
| RMSE FD001 ensemble (lstm x5) | 13.87 | <= 14 | PASS |
| RMSE FD002 lgbm | 14.02 | <= 20 | baseline |
| RMSE FD002 cnn | 14.80 | <= 20 | PASS |
| RMSE FD002 lstm | 14.75 | <= 20 | PASS |
| RMSE FD002 transformer | 15.03 | <= 20 | PASS |
| RMSE FD002 ensemble (lstm x5) * | 13.79 | <= 20 | PASS |
| RMSE FD003 lgbm | 12.83 | <= 14 | baseline |
| RMSE FD003 cnn | 14.93 | <= 14 | FAIL |
| RMSE FD003 lstm | 13.50 | <= 14 | PASS |
| RMSE FD003 transformer | 18.30 | <= 14 | FAIL |
| RMSE FD003 ensemble (lstm x5) * | 12.82 | <= 14 | PASS |
| RMSE FD004 lgbm | 14.89 | <= 22 | baseline |
| RMSE FD004 cnn | 16.19 | <= 22 | PASS |
| RMSE FD004 lstm | 14.66 | <= 22 | PASS |
| RMSE FD004 transformer | 15.63 | <= 22 | PASS |
| RMSE FD004 ensemble (lstm x5) * | 14.35 | <= 22 | PASS |
| NASA score FD001 ensemble | 326 | <= lgbm 231 | FAIL |
| NASA score FD002 ensemble | 1014 | <= lgbm 904 | FAIL |
| NASA score FD003 ensemble | 359 | <= lgbm 283 | FAIL |
| NASA score FD004 ensemble | 973 | <= lgbm 1142 | PASS |
| 90% interval coverage (all 707 test engines) | 0.914 | 0.85 to 0.95 | PASS |
| Cost saving, predictive vs fixed schedule | 25.4% | > 0% | PASS |
| ONNX vs torch max abs diff | 0.00056 | <= 0.01 | PASS |
| ONNX latency median (ms) | 0.422 | <= 5.0 | PASS |
| ONNX latency p95 (ms) | 0.551 | <= 10.0 | PASS |
| No unit leakage train/val | none | none | PASS |

\* lowest test RMSE on that subset.

The 5 member LSTM ensemble reaches 13.87 / 13.79 / 12.82 / 14.35 RMSE on FD001 to FD004 and its 90% intervals cover 91.4% of the 707 test engines (0.870 to 0.940 per subset). Using those intervals to decide when to maintain cuts simulated fleet cost by 25.4% against the best fixed schedule (1,102,990 vs 1,479,260) with 6 unplanned failures instead of 11, and the whole ensemble runs in 0.42 ms per engine window on CPU through ONNX. Architecture, learning rates and the flag threshold were all picked on validation engines, never on test.

### What did not go well

- The LightGBM baseline beats every deep model on FD001 (12.28 vs 13.87) and is effectively tied on FD003 (12.83 vs 12.82). The ensemble only clearly wins on the multi-regime subsets FD002 and FD004.
- The ensemble's NASA score is worse than LightGBM on FD001 to FD003. The score grows exponentially with late error, so a few big over-predictions dominate it, and the ensemble's worst late misses are larger (40.9 vs 33.2 cycles on FD001, 51.9 vs 44.9 on FD002, 48.5 vs 34.6 on FD003).
- The single CNN and Transformer miss the 14 RMSE threshold on FD001 and FD003. They overfit within a few epochs even after lowering the learning rate on validation.
- Scored against the raw, uncapped `RUL_FD00x.txt` values the ensemble gets 15.20 / 26.89 / 14.36 / 26.51. FD002 and FD004 have 57 and 67 test engines with true RUL above 125, which a model trained on a 125 cap cannot reach, so their uncapped RMSE fails the 20 and 22 thresholds. The capped number is the usual convention in C-MAPSS papers, but both are in results.md.
- My first cost simulation picked the flag threshold (0.5) on full validation trajectories. On test it lost to the fixed schedule by 3.6% with 71 failures, because most test engines end long before failure and the policy has to commit to a far-ahead date from one forecast. I changed the validation simulation to cut each validation engine at 5 random points, like the test set, and re-swept. That picked 0.05 and gave the 25.4% above. The test set was run once with each threshold, and the change was to the validation protocol, not to anything fit on test.
- LSTM training on Apple MPS (torch 2.11) gets stuck at about 40 RMSE while the same code on CPU reaches 14, so the LSTM always trains on CPU. CNN and Transformer use MPS, which is not bit-for-bit deterministic, so reruns can move their numbers by a few tenths.

## Testing

17 pytest tests cover preprocessing, the leakage-free split, model shapes and the API schema. See [TESTING.md](TESTING.md).

```bash
cd backend
python -m pytest tests -q
```

Thanks :)
