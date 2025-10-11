# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:12:34 2025

@author: arman
"""

# -*- coding: utf-8 -*-
"""
batch_run_scenarios.py — Parallel scenario runner for Pair_Trade project
Runs multiple simulation configurations using joblib parallelism.
Each scenario’s outputs go into a separate folder under /outputs/.
"""

import os
import shutil
import pandas as pd
from joblib import Parallel, delayed
from simulate_portfolio_son2 import simulate_portfolio
from analyze_portfolio_son import analyze_portfolio

# === Paths ===
BASE_DIR = r"C:\QuantProjects\Pair_Trade"
OUTPUT_BASE = os.path.join(BASE_DIR, "outputs")
WF_FOLDER = os.path.join(BASE_DIR, "data", "wf_pairs")
RATE_CSV = os.path.join(BASE_DIR, "data", "rate_data", "interest_rates_2020_2024.csv")

# === Base PB Config ===
BASE_CONFIG = {
    "starting_capital": 1_000_000.0,
    "years": range(2020, 2025),
    "wf_folder": WF_FOLDER,
    "rate_csv": RATE_CSV,
    "strategy": {"entry_z": 1.5, "exit_z": 0.5},
    "sizing": {
        "fraction": 0.06,
        "target_util": 0.98,
        "min_alloc": 50_000.0,
        "base_z": 1.5,
        "zscale_min": 0.8,
        "zscale_max": 1.4,
    },
    "max_open_positions": 24,
    "fee_rate_roundtrip": 0.0008,
    "slippage_bps": 4.0,
    "short_margin": 0.18,
    "offset_fraction": 0.75,
    "rebate_spread_annual": 0.01,
    "borrow_fee_annual": 0.02,
    "short_cash_credit_fraction": 0.90,
    "debug_leakage_check": True,
}

# === Scenarios (User-Defined) ===
SCENARIOS = {
    # 1️⃣ Base Run
    "base_run": {
        "strategy": {"entry_z": 1.5, "exit_z": 0.5},
        "sizing": {"zscale_min": 0.8, "zscale_max": 1.4},
    },
    # 2️⃣ No scaling (fixed position sizing)
    "zscale_fixed": {
        "strategy": {"entry_z": 1.5, "exit_z": 0.5},
        "sizing": {"zscale_min": 1.0, "zscale_max": 1.0},
    },
    # 3️⃣ Wide scaling range
    "zscale_wide": {
        "strategy": {"entry_z": 1.5, "exit_z": 0.5},
        "sizing": {"zscale_min": 0.5, "zscale_max": 2.0},
    },
    # 4️⃣ Tighter exit, base z-scale
    "exit_tight": {
        "strategy": {"entry_z": 1.5, "exit_z": 0.25},
        "sizing": {"zscale_min": 0.8, "zscale_max": 1.4},
    },
}

# === Core Runner ===
def run_scenario(name: str, overrides: dict):
    print(f"\n[RUN] Scenario: {name}")

    out_dir = os.path.join(OUTPUT_BASE, name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = BASE_CONFIG.copy()
    cfg["strategy"] = {**BASE_CONFIG["strategy"], **overrides.get("strategy", {})}
    cfg["sizing"] = {**BASE_CONFIG["sizing"], **overrides.get("sizing", {})}
    cfg["out_dir"] = out_dir

    try:
        simulate_portfolio(cfg, timeout=60)
        metrics = analyze_portfolio(out_dir, save_plots=False)
        metrics["Run"] = name
        print(f"[OK] {name} done — Final Capital: {metrics['End Capital']:.2f}")
        return metrics
    except Exception as e:
        print(f"[ERR] {name} failed: {e}")
        return {"Run": name, "Error": str(e)}

# === Execute in Parallel ===
if __name__ == "__main__":
    os.chdir(BASE_DIR)
    results = Parallel(n_jobs=-1, backend="loky")(delayed(run_scenario)(n, cfg) for n, cfg in SCENARIOS.items())

    df = pd.DataFrame(results)
    summary_path = os.path.join(OUTPUT_BASE, "run_comparison.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] All scenarios complete. Results saved → {summary_path}")
