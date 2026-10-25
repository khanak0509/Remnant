import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
EVAL_DIR = os.path.join(ROOT, "eval")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MULTI_REGIME = ["FD002", "FD004"]
N_REGIMES = 6

SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS

WINDOW = 30
RUL_CAP = 125
VAL_FRAC = 0.2
STD_TOL = 0.025
SEED = 42
