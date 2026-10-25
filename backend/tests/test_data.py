import numpy as np
import pytest

from config import RUL_CAP, WINDOW
from data import pad, prepare, read, split_units


@pytest.fixture(scope="module")
def fd001():
    return prepare("FD001")


@pytest.fixture(scope="module")
def fd002():
    return prepare("FD002")


def test_no_unit_in_train_and_val(fd001, fd002):
    for d in (fd001, fd002):
        assert set(d["train_units"]).isdisjoint(d["val_units"])
        assert set(np.unique(d["utr"])).isdisjoint(np.unique(d["uval"]))
        assert set(np.unique(d["utr"])) == set(d["train_units"])


def test_split_is_deterministic():
    a = split_units(range(1, 101))
    b = split_units(range(1, 101))
    assert np.array_equal(a[1], b[1]) and len(a[1]) == 20


def test_scaler_fit_on_train_units_only(fd001):
    train = read("FD001")[0]
    tr = train[train["unit"].isin(fd001["train_units"])]
    sc = fd001["scalers"][0]
    assert np.allclose(sc.mean_, tr[fd001["sensors"]].mean().values)
    assert not np.allclose(sc.mean_, train[fd001["sensors"]].mean().values)


def test_regime_scalers(fd002):
    assert len(fd002["scalers"]) == 6
    assert abs(fd002["Xtr"][..., 0].mean()) < 0.1


def test_window_shapes(fd001):
    n = len(fd001["sensors"])
    assert fd001["Xtr"].shape[1:] == (WINDOW, n)
    assert fd001["Xte"].shape == (100, WINDOW, n)
    assert len(fd001["Xtr"]) == len(fd001["ytr"]) == len(fd001["utr"])


def test_rul_cap(fd001):
    assert fd001["ytr"].max() == RUL_CAP
    assert fd001["ytr"].min() == 0
    assert fd001["yte"].max() <= RUL_CAP
    assert fd001["yte_raw"].max() > RUL_CAP


def test_constant_sensors_dropped(fd001):
    assert "s1" in fd001["dropped"] and "s1" not in fd001["sensors"]


def test_pad_short_engine():
    x = np.arange(10, dtype=float).reshape(5, 2)
    p = pad(x)
    assert p.shape == (WINDOW, 2)
    assert np.array_equal(p[0], x[0]) and np.array_equal(p[-5:], x)
