# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:45:56 2025

@author: arman
"""

from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from adaptive_spread_strategy_fixed import compute_rolling_spread, CONFIG as STRAT_DEFAULT

TRADING_DAYS_CAL = 365

# ---------- helpers ----------

def _ann_to_daily(annual_rate: float) -> float:
    return (1.0 + float(annual_rate)) ** (1.0 / TRADING_DAYS_CAL) - 1.0

def _load_rate_schedule(rate_csv: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(rate_csv)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date").drop_duplicates("Date")
    idx = pd.date_range(start=start, end=end, freq="D")
    sdf = pd.DataFrame({"Date": idx.date}).merge(df, on="Date", how="left")
    sdf["Rate"] = sdf["Rate"].ffill().bfill()
    sdf.index = pd.to_datetime(sdf["Date"])
    return sdf[["Rate"]]

def _load_pairs_for_year(wf_folder: str, year: int) -> list[str]:
    f = os.path.join(wf_folder, f"wf_pairs_{year}.csv")
    if not os.path.exists(f):
        return []
    df = pd.read_csv(f)
    if "Pair" in df.columns:
        return df["Pair"].astype(str).tolist()
    return [f"{r['Stock 1']}-{r['Stock 2']}" for _, r in df.iterrows()]

def _cash_required_for_gross(gross_open: float, short_margin: float, offset_fraction: float) -> float:
    long_n  = 0.5 * gross_open
    short_n = 0.5 * gross_open
    base_req = long_n + short_margin * short_n
    offset   = offset_fraction * min(long_n, short_n)
    return max(0.0, base_req - offset)

def _extract_entry_z_cols(df: pd.DataFrame) -> pd.Series:
    candidates = ["EntryZ", "Entry Z", "ZEntry", "z_entry"]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index)

# ---------- core ----------

def simulate_portfolio(config: dict, timeout: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = list(config.get("years", [])) or list(range(2021, 2025))
    starting_capital = float(config["starting_capital"])

    # Costs
    fee_rate_roundtrip = float(config.get("fee_rate_roundtrip", 0.0008))  # 8 bps
    slippage_bps       = float(config.get("slippage_bps", 4.0))          # 4 bps

    # Financing (PB)
    SHORT_MARGIN            = float(config.get("short_margin", 0.18))
    OFFSET_FRACT            = float(config.get("offset_fraction", 0.75))
    REBATE_SPREAD_ANNUAL    = float(config.get("rebate_spread_annual", 0.01))
    BORROW_FEE_ANNUAL       = float(config.get("borrow_fee_annual", 0.02))
    SHORT_CASH_CREDIT_FRAC  = float(config.get("short_cash_credit_fraction", 0.90))

    # Sizing (Z-score magnitude scaling)
    sizing = config.get("sizing", {})
    frac        = float(sizing.get("fraction", 0.06))     # base fraction
    target_util = float(sizing.get("target_util", 0.98))
    min_alloc   = float(sizing.get("min_alloc", 75_000.0))
    max_pos     = int(config.get("max_open_positions", 24))
    base_z      = float(sizing.get("base_z", 1.5))    
    zmin        = float(sizing.get("zscale_min", 0.5))   
    zmax        = float(sizing.get("zscale_max", 2.0))  

    # 1) Trade collection (year-aware, no lookahead)
    trades_df = config.get("trades_df")
    if trades_df is None:
        wf_folder = str(config["wf_folder"])
        user_strat = config.get("strategy", {}) or {}
        strat_cfg = {**(STRAT_DEFAULT or {}), **user_strat}
        all_trades = []
        for y in years:
            start_y = pd.Timestamp(f"{y}-01-01")
            end_y   = pd.Timestamp(f"{y}-12-31")
            pairs_y = _load_pairs_for_year(wf_folder, y)
            if not pairs_y:
                continue
            for pair in pairs_y:
                _, tr_all = compute_rolling_spread(pair, strat_cfg, timeout=timeout)
                if tr_all is None or tr_all.empty:
                    continue
                tr = tr_all.copy()
                tr["Entry Time"] = pd.to_datetime(tr["Entry Time"])
                tr["Exit Time"]  = pd.to_datetime(tr["Exit Time"])
                tr = tr[(tr["Entry Time"] >= start_y) & (tr["Entry Time"] <= end_y)]
                if tr.empty:
                    continue
                tr["Pair"] = str(pair)
                # EntryZ varsa sakla
                tr["EntryZ_"] = _extract_entry_z_cols(tr)
                all_trades.append(tr[["Entry Time","Exit Time","Return","Pair","EntryZ_"]])
        if not all_trades:
            raise ValueError("No trades found. Check wf_folder/years/strategy.")
        trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        trades_df = trades_df.copy()
        if "EntryZ_" not in trades_df.columns:
            trades_df["EntryZ_"] = _extract_entry_z_cols(trades_df)

    # Sanitize + dedupe
    trades_df["Entry Time"] = pd.to_datetime(trades_df["Entry Time"])
    trades_df["Exit Time"]  = pd.to_datetime(trades_df["Exit Time"])
    trades_df = trades_df.sort_values(["Entry Time","Exit Time"]).drop_duplicates(
        subset=["Pair","Entry Time","Exit Time"], keep="first"
    ).reset_index(drop=True)

    # Anti-leakage guard (opsiyonel ama yararlı)
    if config.get("debug_leakage_check", True) and ("wf_folder" in config):
        wf_folder = str(config["wf_folder"])
        allowed_by_year = {y: set(_load_pairs_for_year(wf_folder, y)) for y in years}
        bad = trades_df[
            ~trades_df.apply(lambda r: r["Pair"] in allowed_by_year.get(r["Entry Time"].year, set()), axis=1)
        ]
        if not bad.empty:
            raise AssertionError(
                f"Leakage detected: {len(bad)} unauthorized entries.\n{bad.head(5).to_string(index=False)}"
            )

    # Date index
    start = trades_df["Entry Time"].min().normalize()
    end   = trades_df["Exit Time"].max().normalize()
    daily_idx = pd.date_range(start, end, freq="D")

    # Rates
    rate_csv = str(config["rate_csv"])
    rate_schedule = _load_rate_schedule(rate_csv, start, end)

    # State
    capital = starting_capital
    open_trades: dict[int, dict] = {}
    capital_curve = []
    trade_logs = []

    entries_by_day = trades_df.groupby(trades_df["Entry Time"].dt.date).indices
    exits_by_day   = trades_df.groupby(trades_df["Exit Time"].dt.date).indices

    def gross_open_now() -> float:
        return sum(pos["alloc"] for pos in open_trades.values()) if open_trades else 0.0

    def current_utilization() -> float:
        g = gross_open_now()
        return g / capital if capital > 0 else 0.0

    def room_left() -> float:
        util = current_utilization()
        return max(0.0, target_util - util) * capital

    # Daily loop
    missing_z_count = 0
    for d in daily_idx:
        day_key = d.date()

        # exits
        for i in exits_by_day.get(day_key, []):
            if i not in open_trades:
                continue
            pos = open_trades.pop(i)
            tr  = trades_df.loc[i]
            alloc = pos["alloc"]
            ret = float(tr.get("Return", 0.0))
            alpha_pnl = alloc * ret
            fees = fee_rate_roundtrip * alloc
            slippage = (slippage_bps / 1e4) * alloc
            realized = alpha_pnl - fees - slippage
            capital += realized
            trade_logs.append({
                "Pair": tr["Pair"],
                "Entry Time": pos["entry"],
                "Exit Time": d,
                "Alloc": alloc,
                "EntryZ": pos.get("entryz"),
                "ZScale": pos.get("zscale"),
                "Trade Return": ret,
                "Trade Costs": fees,
                "Slippage": slippage,
                "Alpha PnL": alpha_pnl,
                "Realized PnL": realized,
                "Capital After": capital
            })

        # entries
        for i in entries_by_day.get(day_key, []):
            if len(open_trades) >= max_pos:
                continue
            tr = trades_df.loc[i]
            entry_z = tr.get("EntryZ_", np.nan)
            if pd.isna(entry_z):
                # fallback: no scaling
                z_scale = 1.0
                missing_z_count += 1
            else:
                z_scale = abs(float(entry_z)) / base_z if base_z > 0 else 1.0
                z_scale = max(zmin, min(zmax, z_scale))

            desired = frac * capital * z_scale
            alloc = min(desired, room_left())
            g_now = gross_open_now()
            alloc = min(alloc, max(0.0, capital - g_now))
            if alloc < min_alloc or alloc <= 0.0:
                continue

            open_trades[i] = {"alloc": float(alloc), "entry": d, "pair": tr["Pair"],
                              "entryz": None if pd.isna(entry_z) else float(entry_z),
                              "zscale": float(z_scale)}

        # EOD financing
        g_open = gross_open_now()
        long_n  = 0.5 * g_open
        short_n = 0.5 * g_open

        cash_req = _cash_required_for_gross(g_open, SHORT_MARGIN, OFFSET_FRACT)
        idle_cash = max(0.0, capital - cash_req)

        ann_rate = float(rate_schedule.loc[d, "Rate"]) / 100.0
        deposit_daily = _ann_to_daily(ann_rate)
        rebate_daily  = _ann_to_daily(max(ann_rate - REBATE_SPREAD_ANNUAL, 0.0))
        borrow_daily  = _ann_to_daily(BORROW_FEE_ANNUAL)

        short_cash_collateral = SHORT_CASH_CREDIT_FRAC * short_n
        carry_idle  = idle_cash * deposit_daily
        carry_short = short_cash_collateral * rebate_daily - short_n * borrow_daily

        capital += (carry_idle + carry_short)

        capital_curve.append({
            "Date": d,
            "Capital": capital,
            "Open Gross": g_open,
            "Idle Cash": idle_cash,
            "Open Count": len(open_trades)
        })

    capital_df = pd.DataFrame(capital_curve)
    trade_log_df = pd.DataFrame(trade_logs)

    out_dir = config.get("out_dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        capital_df.to_csv(os.path.join(out_dir, "capital_curve.csv"), index=False)
        trade_log_df.to_csv(os.path.join(out_dir, "trade_log_zmag.csv"), index=False)

    # küçük bilgilendirme
    if missing_z_count > 0:
        print(f"⚠️ EntryZ bulunamadığı için {missing_z_count} işlemde z-scale=1.0 (fallback) kullanıldı.")

    # konsol özeti (hafif)
    if not capital_df.empty:
        final_cap = capital_df["Capital"].iloc[-1]
        start_cap = capital_df["Capital"].iloc[0]
        multiple = final_cap / start_cap
        print(f"\n[OK] Final Capital: {final_cap:,.2f} | Multiple: {multiple:.2f}x")

    return capital_df, trade_log_df


# -------- Example PB config (Z-mag sizing) --------
if __name__ == "__main__":
    CONFIG = {
        "starting_capital": 1_000_000.0,
        "years": range(2020, 2025),
        "wf_folder": r"C:\QuantProjects\Pair_Trade\data\wf_pairs",
        "rate_csv": r"C:\QuantProjects\Pair_Trade\data\rate_data\interest_rates_2020_2024.csv",
        "strategy": {
            "entry_z": 1.5,
            "exit_z": 0.5
        },
        "sizing": {
            "fraction": 0.06,
            "target_util": 0.98,
            "min_alloc": 50_000.0,
            # Z-mag parametreleri:
            "base_z": 1.5,        # referans eşiğin (entry_z) aynısı
            "zscale_min": 0.8,    # alt sınır (çok küçülmesin)
            "zscale_max": 1.4     # üst sınır (çok büyümesin)
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
        "out_dir": r"C:\QuantProjects\Pair_Trade\outputs"
    }

    simulate_portfolio(CONFIG, timeout=60)

