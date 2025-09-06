#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired within-subject stats on MDI deltas.

- Auto-detects k from the MDI file by counting Delta_PC* columns.
- Records choices: k, metric, bootstrap n/seed.
- Computes:
    * mean, SD
    * one-sample t-test vs 0
    * Wilcoxon signed-rank vs 0
    * Cohen's d
    * bootstrap 95% CI for the mean (percentile)
- Saves CSV next to the input MDI file as: mdi_within_stats.csv

Usage:
  python paired_mdi_stats.py --mdi mdi_out/mdi_deltas_PC1..PC3.csv \
      --metric varweighted --boot-n 10000 --boot-seed 7
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple

# ----------------- helpers -----------------

def infer_k_from_mdi_columns(df: pd.DataFrame) -> Optional[int]:
    pcs = []
    for c in df.columns:
        m = re.match(r"Delta_PC(\d+)$", c)
        if m:
            pcs.append(int(m.group(1)))
    return int(max(pcs)) if pcs else None

def bootstrap_ci_mean(x: np.ndarray, n: int = 10000, seed: int = 7) -> Tuple[float, float]:
    """Percentile CI for the mean via bootstrap resampling with replacement."""
    if len(x) < 2 or not np.isfinite(x).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Within-subject stats for MDI deltas (auto-k).")
    ap.add_argument("--mdi", type=Path, required=False, default=Path("mdi_out/mdi_deltas_PC1..PC3.csv"),
                    help="Path to MDI table produced by PCA.py (mdi_deltas_PC1..PCk.csv).")
    ap.add_argument("--metric", choices=["varweighted", "euclid"], default="varweighted",
                    help="Which MDI metric column to analyze.")
    ap.add_argument("--boot-n", type=int, default=10000, help="Number of bootstrap resamples.")
    ap.add_argument("--boot-seed", type=int, default=7, help="Bootstrap RNG seed.")
    args = ap.parse_args()

    df = pd.read_csv(args.mdi)
    # pick metric column
    col_varw = "MDI_VarWeighted_kD"
    col_euc  = "MDI_Euclid_kD"
    if args.metric == "varweighted":
        col = col_varw if col_varw in df.columns else None
    else:
        col = col_euc if col_euc in df.columns else None

    if col is None:
        raise ValueError(f"Metric column not found. Expected '{col_varw}' or '{col_euc}' in {args.mdi}.")

    x = df[col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))

    # infer k from columns
    k = infer_k_from_mdi_columns(df)

    # basic stats
    mean = float(np.mean(x)) if n else float("nan")
    sd   = float(np.std(x, ddof=1)) if n > 1 else float("nan")

    # t-test vs 0
    if n >= 1:
        t_stat, t_p = stats.ttest_1samp(x, 0.0)
        t_stat = float(t_stat); t_p = float(t_p)
    else:
        t_stat = float("nan"); t_p = float("nan")

    # Wilcoxon vs 0
    if n >= 1:
        try:
            w_stat, w_p = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_stat); w_p = float(w_p)
        except ValueError:
            # e.g., all zeros -> Wilcoxon undefined
            w_stat = float("nan"); w_p = float("nan")
    else:
        w_stat = float("nan"); w_p = float("nan")

    # Cohen's d (one-sample)
    d = float(mean / sd) if (n > 1 and sd > 0 and np.isfinite(sd)) else float("nan")

    # bootstrap CI for mean
    ci_low, ci_high = bootstrap_ci_mean(x, n=args.boot_n, seed=args.boot_seed)

    # build output
    out = pd.DataFrame([{
        "k": int(k) if k is not None else np.nan,
        "metric": args.metric,
        "n": n,
        "mean_delta": mean,
        "sd": sd,
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohen_d": d,
        "boot_n": int(args.boot_n),
        "boot_seed": int(args.boot_seed),
        "boot_ci_low": float(ci_low),
        "boot_ci_high": float(ci_high),
    }])

    out_path = args.mdi.with_name("mdi_within_stats.csv")
    out.to_csv(out_path, index=False)
    print("✅ Saved:", out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
