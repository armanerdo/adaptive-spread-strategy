# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:54:29 2025

@author: arman
"""


# generate_all_trades.py — core universe (XU030/XU100 hariç), tek sefer güvenli run
#  - data/daily altındaki tüm hisselerden (XU030, XU100 hariç) pair oluşturur
#  - her pair için trade üretir (entry_z=1.5, exit_z=0.5, confirm_bars=2, timeout=60)
#  - sadece 2018-01-01 ile 2023-12-31 arasında AÇILAN işlemleri tutar (Entry Time filtresi)
#  - chunk'lara yazar, sonunda tek dosyada birleştirir: data/trade_data/all_trades.parquet

import os
import sys
import uuid
from itertools import combinations
import pandas as pd

# --- yerel importlar çalışsın ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adaptive_spread_strategy import compute_rolling_spread, CONFIG as STRAT_DEFAULT  # noqa: E402

# --- klasörler ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DAILY_DIR  = os.path.join(BASE_DIR, "data", "daily")
TRADE_DIR  = os.path.join(BASE_DIR, "data", "trade_data")
os.makedirs(TRADE_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(TRADE_DIR, "all_trades.parquet")
TMP_DIR     = os.path.join(TRADE_DIR, "_tmp_chunks")
os.makedirs(TMP_DIR, exist_ok=True)

# --- tarih penceresi (WF: 2020–2024 için gerekli geçmiş) ---
DATE_START = pd.Timestamp("2018-01-01")
DATE_END   = pd.Timestamp("2023-12-31")  # 2024'te açılan işlemler dahil edilmez

# --- strateji parametreleri (safe, tutarlı) ---
STRATEGY_CONFIG = {
    **STRAT_DEFAULT,      # data_folder vs. burada tanımlı
    "entry_z": 1.5,
    "exit_z": 0.5,
    "confirm_bars": 2,
}
TIMEOUT_BARS = 60

# --- performans/hata kontrolleri ---
CHUNK_SIZE = 75       # her 75 pair'de bir diske yaz
MAX_ERRORS = 150

def list_universe():
    """daily klasöründen ticker listesi üret, XU030/XU100'u hariç tut."""
    files = [f for f in os.listdir(DAILY_DIR) if f.endswith(".parquet")]
    tickers = sorted([f[:-8] for f in files])  # strip ".parquet"
    tickers = [t for t in tickers if t not in {"XU030", "XU100"}]
    if len(tickers) < 2:
        raise RuntimeError("Yetersiz ticker. data/daily altında yeterli parquet yok.")
    return tickers

def flush_buffer(buf, chunk_idx):
    """Buffer'daki trades DF'lerini tek DF yapıp temp'e yazar."""
    if not buf:
        return None
    df = pd.concat(buf, ignore_index=True)
    # çekirdek kolonlar
    keep = [c for c in ["Entry Time", "Exit Time", "Return", "Pair"] if c in df.columns]
    df = df[keep]
    # tip dönüşümleri
    for col in ["Entry Time", "Exit Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    tmp_path = os.path.join(TMP_DIR, f"chunk_{chunk_idx:04d}_{uuid.uuid4().hex[:8]}.parquet")
    df.to_parquet(tmp_path, index=False)
    print(f"💾 Chunk yazıldı: {os.path.basename(tmp_path)} | {len(df)} satır")
    return tmp_path

def combine_chunks(out_file):
    """TMP_DIR altındaki tüm chunk'ları tek dosyada birleştirir."""
    parts = []
    files = sorted([f for f in os.listdir(TMP_DIR) if f.endswith(".parquet")])
    total = 0
    for fn in files:
        fp = os.path.join(TMP_DIR, fn)
        d = pd.read_parquet(fp)
        parts.append(d)
        total += len(d)
    if not parts:
        print("⚠️ Birleştirecek chunk bulunamadı.")
        return 0
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_file, index=False)
    print(f"✅ Final all_trades yazıldı: {out_file} | {len(df)} satır")
    return total

def main():
    # temp temizliği
    for f in os.listdir(TMP_DIR):
        if f.endswith(".parquet"):
            try:
                os.remove(os.path.join(TMP_DIR, f))
            except Exception:
                pass

    tickers = list_universe()
    pairs = [f"{a}-{b}" for a, b in combinations(tickers, 2)]
    print(f"🧮 Ticker sayısı: {len(tickers)} | Pair sayısı: {len(pairs)}")

    buffer = []
    chunk_idx = 0
    errors = 0

    for k, pair in enumerate(pairs, 1):
        try:
            _, trades = compute_rolling_spread(pair, STRATEGY_CONFIG, timeout=TIMEOUT_BARS)
        except Exception as e:
            errors += 1
            print(f"⚠️ {pair}: {e}")
            if errors >= MAX_ERRORS:
                print("❌ Çok fazla hata. Durduruluyor.")
                break
            continue

        if trades is None or trades.empty:
            if k % 50 == 0:
                print(f"... {k}/{len(pairs)} pair işlendi (boş trade).")
            continue

        # --- tarih filtresi (sadece Entry bazlı zorunlu, Exit opsiyonel) ---
        trades = trades.copy()
        trades["Entry Time"] = pd.to_datetime(trades["Entry Time"])
        trades["Exit Time"]  = pd.to_datetime(trades["Exit Time"])
        trades = trades[(trades["Entry Time"] >= DATE_START) & (trades["Entry Time"] <= DATE_END)]
        if trades.empty:
            if k % 50 == 0:
                print(f"... {k}/{len(pairs)} pair işlendi (tarih filtresiyle boş).")
            continue

        trades["Pair"] = pair
        buffer.append(trades[["Entry Time", "Exit Time", "Return", "Pair"]])

        if len(buffer) >= CHUNK_SIZE:
            flush_buffer(buffer, chunk_idx)
            buffer.clear()
            chunk_idx += 1

        if k % 50 == 0:
            print(f"... {k}/{len(pairs)} pair işlendi")

    if buffer:
        flush_buffer(buffer, chunk_idx)

    total = combine_chunks(OUTPUT_FILE)
    if total == 0:
        print("❌ all_trades üretilemedi. data/daily ve config'i kontrol et.")
    else:
        print("🎉 Bitti.")

if __name__ == "__main__":
    main()
