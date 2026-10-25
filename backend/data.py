import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import (COLUMNS, DATA_DIR, MULTI_REGIME, N_REGIMES, RUL_CAP, SEED,
                    SENSORS, SETTINGS, STD_TOL, VAL_FRAC, WINDOW)


def read(subset):
    train = pd.read_csv(f"{DATA_DIR}/train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(f"{DATA_DIR}/test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(f"{DATA_DIR}/RUL_{subset}.txt", header=None).iloc[:, 0].values
    return train, test, rul


def add_rul(df):
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = np.minimum(last - df["cycle"], RUL_CAP)
    return df


def add_test_rul(test, rul):
    last = test.groupby("unit")["cycle"].transform("max")
    final = test["unit"].map(dict(zip(sorted(test["unit"].unique()), rul)))
    test["rul_true"] = final + last - test["cycle"]
    test["rul"] = np.minimum(test["rul_true"], RUL_CAP)
    return test


def split_units(units, frac=VAL_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    units = np.array(sorted(units))
    val = rng.choice(units, size=int(round(len(units) * frac)), replace=False)
    train = np.array([u for u in units if u not in set(val)])
    return train, np.sort(val)


def fit_regimes(df, subset):
    if subset not in MULTI_REGIME:
        return None
    with np.errstate(all="ignore"):
        return KMeans(N_REGIMES, n_init=10, random_state=SEED).fit(df[SETTINGS].values)


def regimes(km, df):
    if km is None:
        return np.zeros(len(df), dtype=int)
    with np.errstate(all="ignore"):
        return km.predict(df[SETTINGS].values)


def pick_sensors(df):
    sd = df.groupby("regime")[SENSORS].std().max()
    keep = [s for s in SENSORS if sd[s] >= STD_TOL]
    drop = [s for s in SENSORS if sd[s] < STD_TOL]
    return keep, drop


def fit_scalers(df, sensors):
    return {r: StandardScaler().fit(g[sensors].values) for r, g in df.groupby("regime")}


def scale(df, scalers, sensors):
    out = df.copy()
    X = df[sensors].values.astype(float)
    for r, sc in scalers.items():
        m = df["regime"].values == r
        if m.any():
            X[m] = sc.transform(X[m])
    out[sensors] = X
    return out


def pad(x, size=WINDOW):
    if len(x) >= size:
        return x
    return np.vstack([np.repeat(x[:1], size - len(x), axis=0), x])


def windows(df, sensors, size=WINDOW):
    X, y, units = [], [], []
    for u, g in df.groupby("unit"):
        x = pad(g[sensors].values, size)
        r = g["rul"].values
        r = np.concatenate([np.full(len(x) - len(r), r[0]), r])
        for end in range(size, len(x) + 1):
            X.append(x[end - size:end])
            y.append(r[end - 1])
            units.append(u)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(units)


def last_windows(df, sensors, size=WINDOW):
    X, y, raw, units = [], [], [], []
    for u, g in df.groupby("unit"):
        X.append(pad(g[sensors].values, size)[-size:])
        y.append(g["rul"].values[-1])
        raw.append(g["rul_true"].values[-1])
        units.append(u)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(raw, dtype=np.float32), np.array(units)


def prepare(subset, verbose=False):
    train, test, rul = read(subset)
    train = add_rul(train)
    test = add_test_rul(test, rul)
    tr_units, val_units = split_units(train["unit"].unique())
    tr = train[train["unit"].isin(tr_units)].copy()
    val = train[train["unit"].isin(val_units)].copy()

    km = fit_regimes(tr, subset)
    for df in (tr, val, test):
        df["regime"] = regimes(km, df)
    sensors, dropped = pick_sensors(tr)
    scalers = fit_scalers(tr, sensors)
    tr, val, test = scale(tr, scalers, sensors), scale(val, scalers, sensors), scale(test, scalers, sensors)

    Xtr, ytr, utr = windows(tr, sensors)
    Xval, yval, uval = windows(val, sensors)
    Xte, yte, yte_raw, ute = last_windows(test, sensors)
    if verbose:
        print(f"{subset}: dropped {dropped}, kept {len(sensors)} sensors")
        print(f"  train {Xtr.shape} val {Xval.shape} test {Xte.shape}")
    return {
        "subset": subset, "sensors": sensors, "dropped": dropped, "kmeans": km, "scalers": scalers,
        "Xtr": Xtr, "ytr": ytr, "utr": utr, "Xval": Xval, "yval": yval, "uval": uval,
        "Xte": Xte, "yte": yte, "yte_raw": yte_raw, "ute": ute, "test_df": test, "train_units": tr_units, "val_units": val_units,
    }


if __name__ == "__main__":
    from config import SUBSETS
    for s in SUBSETS:
        d = prepare(s, verbose=True)
        print(f"  ytr range {d['ytr'].min():.0f}-{d['ytr'].max():.0f}, test true rul {d['yte_raw'].min():.0f}-{d['yte_raw'].max():.0f} ({(d['yte_raw'] > 125).sum()} above cap), units train/val {len(d['train_units'])}/{len(d['val_units'])}")
