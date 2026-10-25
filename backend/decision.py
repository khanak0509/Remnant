import json
import sys

import numpy as np
from scipy.stats import norm

from config import (COST_FAILURE, COST_PLANNED, COST_WASTE, FLAG_PROB, HORIZON, MODEL_DIR, SEED, SUBSETS, WINDOW)
from data import prepare, replay_windows
from uncertainty import best_arch, load_ensemble, predict_dist


def p_fail(mean, sd, horizon=HORIZON):
    return norm.cdf((horizon - mean) / sd)


def cost(m, failure):
    if m >= failure:
        return COST_FAILURE
    return COST_PLANNED + COST_WASTE * (failure - m)


def best_interval(lifetimes):
    grid = np.arange(1, lifetimes.max() + 1)
    totals = [sum(cost(T, f) for f in lifetimes) for T in grid]
    return int(grid[int(np.argmin(totals))])


def predictive_time(mean, sd, cycles, prob=FLAG_PROB, horizon=HORIZON):
    flagged = np.where(p_fail(mean, sd, horizon) > prob)[0]
    if len(flagged):
        return cycles[flagged[0]]
    k = mean[-1] + sd[-1] * norm.ppf(prob) - horizon
    return cycles[-1] + max(0, int(np.ceil(k)))


def replay(members, df, sensors):
    X, units, cycles, rul = replay_windows(df, sensors)
    mean, sd = predict_dist(members, X)
    out = {}
    for u in np.unique(units):
        m = units == u
        out[u] = {"mean": mean[m], "sd": sd[m], "cycles": cycles[m], "failure": int(cycles[m][-1] + rul[m][-1])}
    return out


def truncate(engines, cuts=5, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for u, e in engines.items():
        for k in range(cuts):
            c = rng.integers(WINDOW, e["failure"])
            out[(u, k)] = {"mean": e["mean"][:c], "sd": e["sd"][:c], "cycles": e["cycles"][:c], "failure": e["failure"]}
    return out


def simulate(engines, T, prob=FLAG_PROB):
    totals = {"run_to_failure": 0, "fixed": 0, "predictive": 0}
    stats = {"failures_fixed": 0, "failures_pred": 0, "waste_pred": 0}
    for e in engines.values():
        f = e["failure"]
        m = predictive_time(e["mean"], e["sd"], e["cycles"], prob)
        totals["run_to_failure"] += COST_FAILURE
        totals["fixed"] += cost(T, f)
        totals["predictive"] += cost(m, f)
        stats["failures_fixed"] += int(T >= f)
        stats["failures_pred"] += int(m >= f)
        stats["waste_pred"] += max(0, f - m) if m < f else 0
    return totals, stats


def sweep(subsets):
    arch, _ = best_arch()
    probs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = np.zeros(len(probs))
    fixed = 0
    for s in subsets:
        d = prepare(s)
        members = load_ensemble(arch, s, len(d["sensors"]))
        engines = truncate(replay(members, d["val_df"], d["sensors"]))
        T = best_interval(d["lifetimes"])
        fixed += simulate(engines, T)[0]["fixed"]
        for i, p in enumerate(probs):
            rows[i] += simulate(engines, T, p)[0]["predictive"]
    print(f"val truncated  fixed schedule cost {fixed}")
    for p, c in zip(probs, rows):
        print(f"val truncated  flag prob {p:.2f}  predictive cost {c:.0f}")


def main(subsets):
    arch, _ = best_arch()
    results = {}
    for s in subsets:
        d = prepare(s)
        members = load_ensemble(arch, s, len(d["sensors"]))
        engines = replay(members, d["test_df"], d["sensors"])
        T = best_interval(d["lifetimes"])
        totals, stats = simulate(engines, T)
        saving = 100 * (totals["fixed"] - totals["predictive"]) / totals["fixed"]
        results[s] = {**totals, **stats, "interval": T, "engines": len(engines), "saving_vs_fixed": saving}
        print(f"{s}  engines {len(engines)}  fixed interval {T}  run_to_failure {totals['run_to_failure']}  "
              f"fixed {totals['fixed']}  predictive {totals['predictive']}  saving {saving:.1f}%  "
              f"failures fixed/pred {stats['failures_fixed']}/{stats['failures_pred']}")
    fixed = sum(r["fixed"] for r in results.values())
    pred = sum(r["predictive"] for r in results.values())
    results["all"] = {"fixed": fixed, "predictive": pred, "run_to_failure": sum(r["run_to_failure"] for r in results.values()),
                      "saving_vs_fixed": 100 * (fixed - pred) / fixed}
    print(f"all  fixed {fixed}  predictive {pred}  saving {results['all']['saving_vs_fixed']:.1f}%")
    with open(f"{MODEL_DIR}/results_decision.json", "w") as f:
        json.dump(results, f, indent=2, default=int)


if __name__ == "__main__":
    if sys.argv[1:2] == ["sweep"]:
        sweep(SUBSETS)
    else:
        main(SUBSETS)
