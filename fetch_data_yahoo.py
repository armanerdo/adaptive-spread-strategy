# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:56:29 2025

@author: arman
"""

import os
import yfinance as yf
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

TICKER_FILE = os.path.join(DATA_DIR, "tickers.csv")
OUT_FOLDER  = RAW_DIR

START_DATE = "2016-01-01"
END_DATE   = "2024-12-31"


# === Load ticker list ===
tickers = pd.read_csv(TICKER_FILE)["Ticker"].tolist()

all_data = {}

for ticker in tickers:
    yahoo_ticker = f"{ticker}.IS"
    print(f"[FETCH] {yahoo_ticker}")

    try:
        # yfinance >=0.2.66 default auto_adjust=True
        # Bu yüzden "Close" zaten adj close
        df = yf.download(yahoo_ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"❌ No data for {ticker}")
            continue

        df.index.name = "Date"

        # OHLCV + Close kaydet
        out_path = os.path.join(OUT_FOLDER, f"{ticker}.parquet")
        df.to_parquet(out_path)
        print(f"✅ Saved {out_path}")

        # topluca dataset için Close serisi ekle
        all_data[ticker] = df["Close"]

    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {e}")

# === topluca CSV ===
if all_data:
    adj_df = pd.concat(all_data, axis=1)
    adj_df.to_csv(os.path.join(OUT_FOLDER, "adj_close_all.csv"))
    print("✅ All adjusted close data saved.")
else:
    print("❌ No data fetched. Check ticker list or internet connection.")
