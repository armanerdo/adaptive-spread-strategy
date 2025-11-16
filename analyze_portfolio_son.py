
"""
Final Analysis Module — Pair Trading Project
--------------------------------------------
Analyzes portfolio results, computes full/alpha/carry decomposition,
exports key statistics, and optionally compares multiple runs.
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from datetime import timedelta


# ===== Helpers =====

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _max_drawdown(series: pd.Series) -> float:
    s = series.astype(float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")

def _rolling_sharpe(returns: pd.Series, window: int = 90) -> pd.Series:
    ret = returns.astype(float)
    roll_mean = ret.rolling(window).mean()
    roll_std  = ret.rolling(window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = (roll_mean / roll_std) * np.sqrt(365.0)
    return sr

def _annualize_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(365.0)) if len(daily_returns) else float("nan")

def _cagr(start_cap: float, end_cap: float, days: int) -> float:
    if start_cap <= 0 or end_cap <= 0 or days <= 0:
        return float("nan")
    years = days / 365.0
    return float((end_cap / start_cap) ** (1/years) - 1.0)

def _yearly_returns_from_capital(capital: pd.Series) -> pd.Series:
    cap = capital.sort_index()
    years = sorted({d.year for d in cap.index})
    rets = {}
    for y in years:
        year_slice = cap[(cap.index.year == y)]
        if year_slice.empty:
            continue
        start = year_slice.iloc[0]
        end   = year_slice.iloc[-1]
        if start > 0:
            rets[y] = (end / start) - 1.0
    return pd.Series(rets)

def _compute_alpha_only_curve(trades: pd.DataFrame, start_capital: float) -> pd.Series:
    """Construct capital curve using only realized alpha PnL (no carry)."""
    cap = start_capital
    if trades.empty:
        return pd.Series([], dtype=float)
    idx = pd.date_range(trades["Entry Time"].min().normalize(), trades["Exit Time"].max().normalize(), freq="D")
    series = pd.Series(index=idx, dtype=float)
    series.iloc[0] = cap
    grouped = trades.groupby(trades["Exit Time"].dt.normalize())
    for d in idx[1:]:
        if d in grouped.groups:
            g = trades.loc[grouped.groups[d]]
            pnl = float(g["Realized PnL"].sum()) if "Realized PnL" in g.columns else float((g["Alpha PnL"] - g["Trade Costs"] - g["Slippage"]).sum())
            cap += pnl
        series.loc[d] = cap
    return series

def _plot_series(x, y, title: str, xlabel: str, ylabel: str, out_path: Optional[str] = None) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=140)
    plt.close()

def _plot_hist(data, bins: int, title: str, xlabel: str, ylabel: str, out_path: Optional[str] = None) -> None:
    plt.figure()
    plt.hist(data.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=140)
    plt.close()

def _plot_scatter(x, y, title: str, xlabel: str, ylabel: str, out_path: Optional[str] = None) -> None:
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=140)
    plt.close()


# ===== Main Analysis Function =====

def analyze_portfolio(run_folder: str, save_plots: bool = True) -> Dict[str, Any]:
    cap_path = os.path.join(run_folder, "capital_curve.csv")
    tl_path  = os.path.join(run_folder, "trade_log_zmag.csv")

    if not os.path.exists(cap_path) or not os.path.exists(tl_path):
        raise FileNotFoundError("Both capital_curve.csv and trade_log_zmag.csv must exist in run_folder.")

    capital = pd.read_csv(cap_path, parse_dates=["Date"])
    trades  = pd.read_csv(tl_path, parse_dates=["Entry Time","Exit Time"])

    capital = capital.sort_values("Date").drop_duplicates("Date").set_index("Date")
    cap_series = capital["Capital"].astype(float)
    daily_ret = cap_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    start_cap = float(cap_series.iloc[0])
    end_cap   = float(cap_series.iloc[-1])
    days = int((cap_series.index[-1] - cap_series.index[0]).days + 1)

    # Core metrics
    cagr  = _cagr(start_cap, end_cap, days)
    vol   = _annualize_vol(daily_ret)
    sharpe_full = float((daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(365.0)) if len(daily_ret) else float("nan")
    maxdd = _max_drawdown(cap_series)

    # Alpha-only & carry decomposition
    alpha_curve = _compute_alpha_only_curve(trades, start_capital=start_cap)
    alpha_last = float(alpha_curve.iloc[-1]) if not alpha_curve.empty else start_cap
    carry_contribution = end_cap - alpha_last

    # Sharpe decomposition
    alpha_daily = alpha_curve.pct_change().dropna() if not alpha_curve.empty else pd.Series(dtype=float)
    sharpe_alpha = float((alpha_daily.mean() / alpha_daily.std(ddof=1)) * np.sqrt(365.0)) if len(alpha_daily) else float("nan")
    sharpe_carry = float(sharpe_full - sharpe_alpha) if np.isfinite(sharpe_full) and np.isfinite(sharpe_alpha) else float("nan")

    print("\n📊 Sharpe Decomposition")
    print(f"  • Full Portfolio Sharpe : {sharpe_full:.3f}")
    print(f"  • Alpha-Only Sharpe     : {sharpe_alpha:.3f}")
    print(f"  • Carry-Only (Diff)     : {sharpe_carry:.3f}")

    yr = _yearly_returns_from_capital(cap_series)

    # === Plots ===
    plots_dir = os.path.join(run_folder, "plots")
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        _plot_series(cap_series.index, cap_series.values, "Capital Curve", "Date", "Capital",
                     os.path.join(plots_dir, "capital_curve.png"))
        _plot_series(cap_series.index, np.log(cap_series.values), "Capital Curve (log)", "Date", "log(Capital)",
                     os.path.join(plots_dir, "capital_curve_log.png"))

        peak = cap_series.cummax()
        dd = (cap_series / peak) - 1.0
        _plot_series(dd.index, dd.values, "Drawdown", "Date", "Drawdown",
                     os.path.join(plots_dir, "drawdown.png"))

        rs = _rolling_sharpe(daily_ret, window=90)
        _plot_series(rs.index, rs.values, "Rolling Sharpe (90d)", "Date", "Sharpe",
                     os.path.join(plots_dir, "rolling_sharpe_90d.png"))

        if "Open Count" in capital.columns:
            _plot_series(capital.index, capital["Open Count"].values, "Open Positions Over Time", "Date", "Open Count",
                         os.path.join(plots_dir, "open_positions.png"))

        if "Open Gross" in capital.columns:
            util = (capital["Open Gross"] / cap_series).replace([np.inf, -np.inf], np.nan)
            _plot_series(capital.index, util.values, "Utilization (Open Gross / Capital)", "Date", "Utilization",
                         os.path.join(plots_dir, "utilization.png"))

        if "Trade Return" in trades.columns and not trades.empty:
            _plot_hist(trades["Trade Return"], bins=50,
                       title="Trade Return Distribution", xlabel="Return", ylabel="Count",
                       out_path=os.path.join(plots_dir, "trade_returns_hist.png"))

        if "EntryZ" in trades.columns and trades["EntryZ"].notna().any():
            _plot_scatter(trades["EntryZ"], trades["Trade Return"],
                          "Entry Z vs Trade Return", "Entry Z", "Trade Return",
                          os.path.join(plots_dir, "entryz_vs_return.png"))

        if len(yr):
            plt.figure()
            yr.sort_index().plot(kind="bar")
            plt.title("Annual Returns")
            plt.xlabel("Year")
            plt.ylabel("Return")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "annual_returns.png"), dpi=140)
            plt.close()

    # === Top & Bottom Trades ===
    if not trades.empty and "Trade Return" in trades.columns:
        top = trades.sort_values("Trade Return", ascending=False).head(10)
        worst = trades.sort_values("Trade Return", ascending=True).head(10)
        top.to_csv(os.path.join(run_folder, "top_10_trades.csv"), index=False)
        worst.to_csv(os.path.join(run_folder, "worst_10_trades.csv"), index=False)

    # === Trade Stats ===
    trade_stats = {}
    if not trades.empty and "Trade Return" in trades.columns:
        tr = trades["Trade Return"].dropna()
        trade_stats = {
            "trades": int(len(trades)),
            "win_rate": float((tr > 0).mean()) if len(tr) else float("nan"),
            "avg_return": float(tr.mean()) if len(tr) else float("nan"),
            "std_return": float(tr.std(ddof=1)) if len(tr) else float("nan"),
            "avg_win": float(tr[tr > 0].mean()) if (tr > 0).any() else float("nan"),
            "avg_loss": float(tr[tr <= 0].mean()) if (tr <= 0).any() else float("nan"),
            "median_return": float(tr.median()) if len(tr) else float("nan"),
            "max_return": float(tr.max()) if len(tr) else float("nan"),
            "min_return": float(tr.min()) if len(tr) else float("nan"),
        }

    metrics: Dict[str, Any] = {
        "Start Capital": start_cap,
        "End Capital": end_cap,
        "Multiple": end_cap / start_cap if start_cap > 0 else float("nan"),
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe_full,
        "Sharpe Full": sharpe_full,
        "Sharpe Alpha": sharpe_alpha,
        "Sharpe Carry": sharpe_carry,
        "Max Drawdown": maxdd,
        "AlphaOnly End": alpha_last,
        "Carry Contribution": carry_contribution,
        "Annual Returns": {int(k): float(v) for k, v in yr.to_dict().items()},
        "Trade Stats": trade_stats,
        "Days": days,
        "First Date": str(cap_series.index[0].date()),
        "Last Date": str(cap_series.index[-1].date())
    }

    return metrics


# ===== Batch Comparison Mode =====

def compare_runs(run_folders: list[str]) -> pd.DataFrame:
    """Compare multiple scenario folders and export summary CSV."""
    results = []
    for folder in run_folders:
        try:
            m = analyze_portfolio(folder, save_plots=False)
            results.append({
                "Run": os.path.basename(folder),
                **{k: v for k, v in m.items() if isinstance(v, (float, int))}
            })
        except Exception as e:
            print(f"[WARN] {folder}: {e}")
    df = pd.DataFrame(results)
    out_path = os.path.join("outputs", "run_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Comparison saved → {out_path}")
    return df



if __name__ == "__main__":
    os.chdir(r"C:\QuantProjects\Pair_Trade")
    metrics = analyze_portfolio(run_folder="outputs", save_plots=True)
    print(metrics)
    # compare_runs(["outputs/base_run", "outputs/strict_z", "outputs/zscale_tight"])



