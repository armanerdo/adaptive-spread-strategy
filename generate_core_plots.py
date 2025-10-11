# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 00:27:24 2025

@author: arman
"""

# ===============================================
# generate_core_plots.py
# Creates core visuals for Adaptive Spread Strategy report
# ===============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- Config ----------
RUN_FOLDER = r"C:\QuantProjects\Pair_Trade\outputs\base_run"
PLOT_DIR = os.path.join(RUN_FOLDER, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------- Load Data ----------
cap_path = os.path.join(RUN_FOLDER, "capital_curve.csv")
capital = pd.read_csv(cap_path, parse_dates=["Date"]).sort_values("Date")

cap = capital["Capital"].astype(float)
cap.index = capital["Date"]

# Daily returns & rolling Sharpe
daily_ret = cap.pct_change().dropna()
rolling_sharpe = (daily_ret.rolling(90).mean() /
                  daily_ret.rolling(90).std(ddof=1)) * np.sqrt(365)

# ---------- 1. Capital Curve ----------
plt.figure(figsize=(10, 5))
plt.plot(cap.index, cap.values, lw=2, color="navy", label="Portfolio Value (₺)")
plt.title("Capital Curve (Base Run)", fontsize=13)
plt.xlabel("Date"); plt.ylabel("Capital (₺)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "capital_curve.png"), dpi=250)

# ---------- 2. Drawdown Curve ----------
peak = cap.cummax()
drawdown = (cap / peak) - 1.0
plt.figure(figsize=(10, 4))
plt.fill_between(drawdown.index, drawdown.values, 0, color="salmon", alpha=0.6)
plt.title("Portfolio Drawdown", fontsize=13)
plt.xlabel("Date"); plt.ylabel("Drawdown")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "drawdown.png"), dpi=250)

# ---------- 3. Rolling 90-Day Sharpe ----------
plt.figure(figsize=(10, 4))
plt.plot(rolling_sharpe.index, rolling_sharpe.values, color="black", lw=1.5)
plt.title("Rolling 90-Day Sharpe Ratio", fontsize=13)
plt.xlabel("Date"); plt.ylabel("Sharpe")
plt.axhline(0, color="gray", lw=0.8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rolling_sharpe_90d.png"), dpi=250)

# ---------- 4. Annual Returns (fixed to include 2020) ----------
cap_df = pd.DataFrame({"Capital": cap})
cap_df["Year"] = cap_df.index.year
annual_returns = (
    cap_df.groupby("Year")["Capital"].apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
)

plt.figure(figsize=(6, 4))
plt.bar(annual_returns.index, annual_returns.values * 100,
        color="steelblue", width=0.5)
plt.title("Annual Portfolio Returns", fontsize=13)
plt.xlabel("Year"); plt.ylabel("Return (%)")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "annual_returns.png"), dpi=250)


print(f"✅ Core plots saved to: {PLOT_DIR}")

