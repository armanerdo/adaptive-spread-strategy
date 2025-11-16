# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 00:33:27 2025

@author: arman
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

CONFIG = {
    "data_folder": r"C:\QuantProjects\Pair_Trade\data\daily",
    "entry_z": 1.5,
    "exit_z": 0.5
}

# ----------------------------
# Helpers
# ----------------------------

def _compute_beta_series_fixed(merged, window=90):
    y = merged["logP1"].values
    x = merged["logP2"].values
    betas = np.full(len(merged), np.nan, dtype=float)
    for i in range(len(merged)):
        if i < window:
            continue
        Xi = x[i - window:i].reshape(-1, 1)
        yi = y[i - window:i]
        betas[i] = LinearRegression().fit(Xi, yi).coef_[0]
    return pd.Series(betas, index=merged.index)


def _spread_from_beta(merged, beta_series):
    return merged["logP1"] - beta_series * merged["logP2"]


def _causal_spread_vol(spread, lookback=60, ema_alpha=0.2):
    dS = spread.diff()
    vol_raw = dS.rolling(lookback, min_periods=lookback).std()
    vol = vol_raw.ewm(alpha=ema_alpha, adjust=False).mean()
    return vol


def _map_vol_to_windows_series(vol_series,
                               beta_min=30, beta_max=90,
                               z_min=15, z_max=30):
    hist = []
    beta_ws, z_ws = [], []
    for v in vol_series:
        if np.isnan(v):
            beta_ws.append(beta_min)
            z_ws.append(z_min)
            continue
        hist.append(v)
        pct = np.sum(np.array(hist) <= v) / len(hist)
        beta_w = int(beta_min + pct * (beta_max - beta_min))
        z_w    = int(z_min   + pct * (z_max   - z_min))
        beta_ws.append(beta_w)
        z_ws.append(z_w)
    return pd.Series(beta_ws, index=vol_series.index), pd.Series(z_ws, index=vol_series.index)


def _compute_beta_series_adaptive(merged, beta_w_series, default_w=90):
    y = merged["logP1"].values
    x = merged["logP2"].values
    betas = np.full(len(merged), np.nan, dtype=float)
    for i in range(len(merged)):
        w = beta_w_series.iloc[i - 1] if i > 0 else pd.NA
        w = int(w) if pd.notna(w) else default_w
        if i < w:
            continue
        Xi = x[i - w:i].reshape(-1, 1)
        yi = y[i - w:i]
        betas[i] = LinearRegression().fit(Xi, yi).coef_[0]
    return pd.Series(betas, index=merged.index)


def _zscore_causal(spread, z_w_series, min_window=20, eps=1e-9):
    z = pd.Series(np.nan, index=spread.index)
    mu_series = pd.Series(np.nan, index=spread.index)
    sd_series = pd.Series(np.nan, index=spread.index)
    for i in range(len(spread)):
        w = z_w_series.iloc[i]
        if pd.isna(w):
            continue
        w = int(max(int(w), min_window))
        if i < w:
            continue
        win = spread.iloc[i - w:i]
        mu = win.mean()
        sd = win.std(ddof=1)
        if not np.isfinite(sd) or sd < eps:
            continue
        z.iloc[i] = (spread.iloc[i] - mu) / sd
        mu_series.iloc[i] = mu
        sd_series.iloc[i] = sd
    return z, mu_series, sd_series

# ----------------------------
# Main
# ----------------------------

def compute_rolling_spread(pair, config=CONFIG, *,
                           base_beta_window=90,
                           vol_lookback=60,
                           ema_alpha=0.2,
                           timeout=None):

    t1, t2 = pair.split("-")

    df1 = pd.read_parquet(os.path.join(config["data_folder"], f"{t1}.parquet"))
    df2 = pd.read_parquet(os.path.join(config["data_folder"], f"{t2}.parquet"))

    # ✅ Use rolling_2 style data handling (dropna only)
    merged = pd.DataFrame(index=df1.index.union(df2.index)).sort_index()
    merged["P1"] = df1["Close"]
    merged["P2"] = df2["Close"]
    merged.dropna(inplace=True)

    merged["logP1"] = np.log(merged["P1"])
    merged["logP2"] = np.log(merged["P2"])

    # 1) Bootstrap β with fixed window (causal)
    beta_base = _compute_beta_series_fixed(merged, window=base_beta_window)
    spread_base = _spread_from_beta(merged, beta_base)

    # 2) Causal realized vol of Δspread (smoothed)
    vol = _causal_spread_vol(spread_base, lookback=vol_lookback, ema_alpha=ema_alpha)

    # 3) Map vol_t → per-bar windows
    beta_w_series, z_w_series = _map_vol_to_windows_series(vol)

    # 4) Adaptive β_t using lagged beta_window
    beta_adapt = _compute_beta_series_adaptive(merged, beta_w_series, default_w=base_beta_window)
    merged["beta"] = beta_adapt

    # 5) Spread with adaptive β_t
    merged["spread"] = _spread_from_beta(merged, beta_adapt)

    # 6) Causal z-score
    z, mu, sd = _zscore_causal(merged["spread"], z_w_series)
    merged["zscore"] = z
    merged["spread_mean"] = mu
    merged["spread_std"]  = sd
    merged["z_w_used"]    = z_w_series

    # 7) Entry/exit logic (same as before)
    position = 0
    entry_idx = None
    trade_log = []
    signal_series = []

    for i in range(len(merged)):
        z = merged["zscore"].iloc[i]
        if np.isnan(z):
            signal_series.append(position if position != 0 else 0)
            continue

        if position == 0 and i >= 1:
            prev_z = merged["zscore"].iloc[i - 1]
            if (z > config["entry_z"]) and (prev_z > config["entry_z"]):
                position = -1
                entry_idx = i
            elif (z < -config["entry_z"]) and (prev_z < -config["entry_z"]):
                position = 1
                entry_idx = i
        else:
            exit_now = (abs(z) < config["exit_z"])
            if (timeout is not None) and (entry_idx is not None):
                exit_now = exit_now or (i - entry_idx >= timeout)

            if exit_now and entry_idx is not None:
                row = merged.iloc[i]
                ent = merged.iloc[entry_idx]
                r1 = np.log(row["P1"] / ent["P1"])
                r2 = np.log(row["P2"] / ent["P2"])
                trade_return = position * (r1 - ent["beta"] * r2)
                trade_log.append({
                    "Entry Time": ent.name,
                    "Exit Time": row.name,
                    "Position": position,
                    "Entry Z": ent["zscore"],
                    "Exit Z": z,
                    "Beta@Entry": ent["beta"],
                    "Return": trade_return,
                    "LongLeg": t1 if position == 1 else t2,
                    "ShortLeg": t2 if position == 1 else t1
                })
                position = 0
                entry_idx = None

        signal_series.append(position)

    merged["signal"] = signal_series
    trades = pd.DataFrame(trade_log)
    return merged, trades


if __name__ == "__main__":
    pair = "AEFES-AKSEN"
    df, trades = compute_rolling_spread(pair, CONFIG, timeout=60)
    print("\n📊 Sample Trades")
    print(trades.head())

