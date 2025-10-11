# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 17:48:40 2025

@author: arman
"""

# -*- coding: utf-8 -*-
"""
Rebuild run_comparison.csv from existing scenario outputs
(No re-simulation needed)
"""

import os
import sys
import pandas as pd

# === Paths ===
BASE_DIR = r"C:\QuantProjects\Pair_Trade"
OUTPUT_BASE = os.path.join(BASE_DIR, "outputs")

# make sure local modules are importable
sys.path.append(BASE_DIR)

from analyze_portfolio_son import analyze_portfolio

# === Find valid scenario folders ===
folders = [
    os.path.join(OUTPUT_BASE, f)
    for f in os.listdir(OUTPUT_BASE)
    if os.path.isdir(os.path.join(OUTPUT_BASE, f))
    and os.path.exists(os.path.join(OUTPUT_BASE, f, "capital_curve.csv"))
]

print(f"🔍 Found {len(folders)} scenario folders.")

results = []
for folder in folders:
    name = os.path.basename(folder)
    try:
        metrics = analyze_portfolio(folder, save_plots=False)
        metrics["Run"] = name
        print(f"[OK] {name} analyzed successfully.")
        results.append(metrics)
    except Exception as e:
        print(f"[ERR] {name}: {e}")

# === Save rebuilt comparison ===
if results:
    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_BASE, "run_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Rebuilt comparison saved → {out_path}")

    # --- Optional summary printout ---
    cols = ["Run", "End Capital", "CAGR", "Sharpe", "Max Drawdown"]
    available = [c for c in cols if c in df.columns]
    if available:
        print("\n📊 Summary:")
        print(df[available].sort_values("Sharpe", ascending=False).to_string(index=False))
else:
    print("⚠️ No valid runs found.")
