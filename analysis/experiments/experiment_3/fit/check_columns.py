#!/usr/bin/env python3
"""Quick script to check column names in CmdStanPy summary."""

import json
from pathlib import Path

from shared_utils import compile_model, fit_model

BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis")
MODEL_PATH = BASE_DIR / "experiments" / "experiment_3" / "model.stan"
DATA_PATH = BASE_DIR / "data" / "stan_data.json"

with open(DATA_PATH) as f:
    data = json.load(f)

model = compile_model(MODEL_PATH)
fit = fit_model(model, data, chains=2, warmup=50, sampling=50, show_progress=False)

summary = fit.summary()
print("Column names:")
print(summary.columns.tolist())
