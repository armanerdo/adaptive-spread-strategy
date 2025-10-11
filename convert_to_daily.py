# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:00:39 2025

@author: arman
"""

# convert_to_daily.py
# Converts raw Yahoo Finance parquet files (from fetch_data_yahoo.py)
# into clean daily OHLC parquet files for use by the adaptive spread strategy.

import os
import pandas as pd

# === Directory setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DAILY_DIR = os.path.join(BASE_DIR, "data", "daily")
os.makedirs(DAILY_DIR, exist_ok=True)

# === Conversion loop ===
for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".parquet"):
        continue

    ticker = filename.replace(".parquet", "")
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_parquet(path)

    # Keep consistent OHLCV structure
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Save cleaned file
    out_path = os.path.join(DAILY_DIR, f"{ticker}.parquet")
    df.to_parquet(out_path)
    print(f"✅ Converted {ticker} → {out_path}")
