
# design_and_build_wf_pairs.py (CAUSAL ROLLING)
# ---------------------------------------------
# Walk-forward pair selection with STRICTLY CAUSAL priors and ban list.
# For each year Y, we ONLY use trades whose Entry Time is before Y-01-01
# and within a 2-year lookback window to:
#   - compute per-pair EMA mean/std → Quality Score (QS)
#   - compute a per-year ban list from bottom-quantile SharpeLike
#   - compute per-year priors (same-sector tilt, sector-pair whitelist)
# Then we rank by QS (+ small prior boosts) and take TOP_N pairs.
#
# Usage: python design_and_build_wf_pairs.py
# Outputs: wf_pairs_YYYY.csv in OUT_DIR + ban_list_YYYY.csv (for audit).

# walkforward_pair_selection.py — Pair_Trade için güncel sürüm
# Causal (lookback 2 yıl) walk-forward pair selection
# Input : data/trade_data/all_trades.parquet
# Output: data/wf_pairs/wf_pairs_YYYY.csv ve ban_list_YYYY.csv

import os, math
import pandas as pd, numpy as np
from pathlib import Path

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_TRADES_PARQUET = os.path.join(BASE_DIR, "data", "trade_data", "all_trades.parquet")
TRADE_LOG_CSV      = None
OUT_DIR            = os.path.join(BASE_DIR, "data", "wf_pairs")
os.makedirs(OUT_DIR, exist_ok=True)

# === CONFIG ===
YEARS_START = 2020        # 2020–2024 arası
YEARS_END   = 2024
LOOKBACK_DAYS = 730       # 2 yıl
MIN_TRADES   = 3
EMA_ALPHA    = 0.20
TOP_N        = 25
MIN_TRADES_SECPAIR = 20   # sektör-pair whitelist

# Ban list parametreleri
BAN_MIN_TRADES   = 5
BAN_QUANTILE     = 0.20
BAN_REQUIRE_NONPOS_MEAN = True

# === SEKTÖR MAP ===
SECTOR_MAP = {
    "AKBNK":"Bank","GARAN":"Bank","YKBNK":"Bank","VAKBN":"Bank","ISCTR":"Bank","HALKB":"Bank","TSKB":"Bank",
    "KCHOL":"Hold","SAHOL":"Hold","DOHOL":"Hold","ENKAI":"Hold",
    "THYAO":"Air","TAVHL":"Air","PGSUS":"Travel",
    "BIMAS":"Retail","MGROS":"Retail","SOKM":"Retail",
    "SISE":"Glass","KRDMD":"Steel","CIMSA":"Cement","OYAKC":"Cement","FROTO":"Auto","TOASO":"Auto",
    "ARCLK":"Durables","VESTL":"Durables","TKFEN":"Constr","ENJSA":"Energy",
    "TCELL":"Telecom","PETKM":"Chem","ALARK":"Invest","ODAS":"Energy","EKGYO":"RE","TUPRS":"Refinery",
    "SASA":"Chem","HEKTS":"Agri","ZOREN":"Energy","ASELS":"Defence","DOAS":"Auto","KONTR":"Tech"
}

# === HELPERS ===
def split_pair(p):
    if isinstance(p, str) and "-" in p:
        a, b = p.split("-", 1)
        return a.strip(), b.strip()
    return None, None

def sec_key(a,b):
    x = sorted([a,b])
    return f"{x[0]}-{x[1]}"

def load_trades(input_parquet: str | None, input_csv: str | None):
    if input_parquet and os.path.exists(input_parquet):
        df = pd.read_parquet(input_parquet)
    elif input_csv and os.path.exists(input_csv):
        df = pd.read_csv(input_csv, parse_dates=["Entry Time","Exit Time"])
    else:
        raise FileNotFoundError("Provide ALL_TRADES_PARQUET or TRADE_LOG_CSV path")

    if "Return" not in df.columns and "Realized PnL" in df.columns and "Alloc" in df.columns:
        df["Return"] = df["Realized PnL"] / df["Alloc"]

    for col in ["Entry Time","Exit Time"]:
        df[col] = pd.to_datetime(df[col])
    if "Pair" not in df.columns:
        raise ValueError("Trades file must contain a 'Pair' column")

    df = df.dropna(subset=["Pair","Entry Time","Exit Time","Return"])
    return df

def compute_year_priors(hist: pd.DataFrame, min_trades_secpair: int = 20):
    """Compute priors using ONLY lookback history."""
    if hist.empty:
        return {"prefer_same_sector": False, "sectorpair_whitelist": []}

    A,B = zip(*hist["Pair"].map(split_pair))
    secA = pd.Series(A).map(SECTOR_MAP).fillna("Other")
    secB = pd.Series(B).map(SECTOR_MAP).fillna("Other")
    same = (secA == secB).astype(int)
    secpair = [sec_key(a,b) for a,b in zip(secA, secB)]
    h = hist.copy()
    h["SameSector"] = same
    h["SectorPair"] = secpair

    # same vs cross
    by_same = h.groupby("SameSector").agg(MeanRet=("Return","mean")).reset_index()
    mean_ret_same  = by_same.loc[by_same["SameSector"]==1,"MeanRet"].mean() if (by_same["SameSector"]==1).any() else 0.0
    mean_ret_cross = by_same.loc[by_same["SameSector"]==0,"MeanRet"].mean() if (by_same["SameSector"]==0).any() else 0.0
    prefer_same = (mean_ret_same > mean_ret_cross)

    # sector-pair whitelist
    by_secp = h.groupby("SectorPair").agg(Trades=("Return","size"), MeanRet=("Return","mean")).reset_index()
    whitelist = by_secp[by_secp["Trades"]>=min_trades_secpair].sort_values("MeanRet",ascending=False)
    secpair_whitelist = whitelist[whitelist["MeanRet"]>0]["SectorPair"].head(8).tolist()

    return {"prefer_same_sector": prefer_same, "sectorpair_whitelist": secpair_whitelist}

def compute_ban_list(hist_pair_stats: pd.DataFrame):
    stable = hist_pair_stats[hist_pair_stats["Trades"]>=BAN_MIN_TRADES].copy()
    stable = stable.replace([np.inf,-np.inf],np.nan).dropna(subset=["SharpeLike"])
    if stable.empty: 
        return []
    q_thr = float(stable["SharpeLike"].quantile(BAN_QUANTILE))
    if BAN_REQUIRE_NONPOS_MEAN:
        banned = stable[(stable["SharpeLike"]<=q_thr) & (stable["MeanRet"]<=0)]["Pair"].tolist()
    else:
        banned = stable[stable["SharpeLike"]<=q_thr]["Pair"].tolist()
    return banned

# === CORE WF BUILDER ===
def build_wf_pairs_causal(full_df: pd.DataFrame, years, lookback_days, min_trades, ema_alpha, topN, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    full_df = full_df.copy().sort_values(["Entry Time","Exit Time"])

    # Sector annotations
    A_all,B_all = zip(*full_df["Pair"].map(split_pair))
    full_df["SectorA"] = pd.Series(A_all).map(SECTOR_MAP).fillna("Other")
    full_df["SectorB"] = pd.Series(B_all).map(SECTOR_MAP).fillna("Other")
    full_df["SameSector"] = (full_df["SectorA"]==full_df["SectorB"]).astype(int)
    full_df["SectorPair"] = [sec_key(a,b) for a,b in zip(full_df["SectorA"], full_df["SectorB"])]

    for y in years:
        asof = pd.Timestamp(f"{y}-01-01")
        lb   = asof - pd.Timedelta(days=lookback_days)
        hist = full_df[(full_df["Entry Time"]<asof) & (full_df["Entry Time"]>=lb)].copy()
        if hist.empty:
            pd.DataFrame(columns=["Stock 1","Stock 2"]).to_csv(Path(out_dir)/f"wf_pairs_{y}.csv", index=False)
            (Path(out_dir)/f"ban_list_{y}.csv").write_text("Pair\n")
            print(f"{y}: no history, wrote empty files")
            continue

        # ---- priors ----
        priors_y = compute_year_priors(hist, min_trades_secpair=MIN_TRADES_SECPAIR)
        pref_same = priors_y.get("prefer_same_sector", False)
        whitelist = set(priors_y.get("sectorpair_whitelist", []))

        # ---- ban list ----
        pair_stats = (hist.groupby("Pair")
                         .agg(Trades=("Return","size"),
                              MeanRet=("Return","mean"),
                              StdRet=("Return","std"))
                         .reset_index())
        pair_stats["SharpeLike"] = pair_stats["MeanRet"] / pair_stats["StdRet"]
        ban_list_y = compute_ban_list(pair_stats)
        pd.DataFrame({"Pair": ban_list_y}).to_csv(Path(out_dir)/f"ban_list_{y}.csv", index=False)

        # ---- EMA stats ----
        stats = []
        for pair, g in hist.groupby("Pair"):
            if pair in set(ban_list_y):
                continue
            m=v=0.0; n=0
            for r in g["Return"]:
                if n==0: m=r; v=0.0; n=1
                else:
                    m=(1-ema_alpha)*m + ema_alpha*r
                    v=(1-ema_alpha)*v + ema_alpha*(r - m)**2
                    n+=1
            if n>=min_trades:
                stats.append((pair, m, math.sqrt(max(v,1e-12)), n))
        if not stats:
            pd.DataFrame(columns=["Stock 1","Stock 2"]).to_csv(Path(out_dir)/f"wf_pairs_{y}.csv", index=False)
            print(f"{y}: no eligible pairs after bans/min_trades")
            continue

        st = pd.DataFrame(stats, columns=["Pair","MeanEMA","StdEMA","N"])
        st["QS"] = st["MeanEMA"] / st["StdEMA"]
        st = st.merge(hist[["Pair","SameSector","SectorPair"]].drop_duplicates("Pair"), on="Pair", how="left")

        # Prior boosts
        if pref_same:
            st["QS"] += 0.10 * st["SameSector"].astype(float)
        if whitelist:
            st["QS"] += st["SectorPair"].isin(whitelist).astype(int) * 0.10

        st = st.sort_values("QS", ascending=False).head(topN)
        stocks = st["Pair"].str.split("-", n=1, expand=True)
        wf = pd.DataFrame({"Stock 1": stocks[0], "Stock 2": stocks[1]})
        wf.to_csv(Path(out_dir)/f"wf_pairs_{y}.csv", index=False)
        print(f"{y}: wrote {len(wf)} pairs (causal)")

# === MAIN ===
if __name__ == "__main__":
    df = load_trades(ALL_TRADES_PARQUET, TRADE_LOG_CSV)
    years = range(YEARS_START, YEARS_END + 1)
    build_wf_pairs_causal(df, years, LOOKBACK_DAYS, MIN_TRADES, EMA_ALPHA, TOP_N, OUT_DIR)
