# Testing

There are two layers: unit tests with pytest, and the full evaluation script that scores every stage against a threshold.

## Unit tests

```bash
cd backend
python -m pytest tests -q
```

The data and model tests only need the raw files in `backend/data/`. The API tests need the exported models (`python export.py`), since the app loads the ONNX ensembles and preprocessing on startup.

| file | what it checks |
|------|----------------|
| tests/test_data.py | no engine unit in both train and val, split is deterministic, scalers fit on train units only, 6 regime scalers for FD002, window shapes, RUL cap at 125, constant sensors dropped, short engines padded to 30 cycles |
| tests/test_models.py | CNN, LSTM and Transformer forward pass shapes for 1 output (point) and 2 outputs (mean and variance) |
| tests/test_api.py | /predict response schema and interval ordering, short windows get padded, /fleet/stream emits cycles in order with the right fields, /fleet/status counts |

## Evaluation

```bash
cd backend
python eval/run_all.py
```

This reads the results saved by each stage (baseline.py, train.py, uncertainty.py, decision.py, export.py), re-runs the leakage check across all four subsets, prints one table with metric, result, threshold and status, and writes the per-subset tables to `backend/eval/results.md`.

Thresholds:

| metric | threshold |
|--------|-----------|
| RMSE FD001 / FD003 | <= 14 |
| RMSE FD002 | <= 20 |
| RMSE FD004 | <= 22 |
| NASA score | <= LightGBM baseline on the same subset |
| 90% interval coverage | 0.85 to 0.95 |
| cost saving vs fixed schedule | > 0% |
| ONNX latency median / p95 | <= 5 ms / <= 10 ms |
| ONNX vs PyTorch output | max abs diff <= 0.01 |
| unit leakage | none |

LightGBM rows are marked `baseline` and are not scored against the threshold. Nothing in the evaluation is tuned on the test set: architecture, learning rates and the flag threshold were chosen on the validation engines.

## Manual check

```bash
cd backend && uvicorn main:app --port 8000
cd frontend && npm run dev
```

Open http://localhost:5173, press Start replay, and check that cells move from OK to Watch to Maintain as engines approach the end of their test trajectories, and that clicking a cell shows its predicted and true RUL.
