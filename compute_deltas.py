#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute First→Last deltas for selected feature families.

Outputs (per family):
  mdi_out/deltas_STD.csv
  mdi_out/deltas_CV.csv
  mdi_out/deltas_ZeroCross.csv

If --plot is set, also saves PNG heatmaps for quick inspection.

Usage:
  python compute_deltas.py --unified unified_features.csv --outdir mdi_out \
      --include "*_STD,*_CV,*_ZeroCross" --plot
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- parsing helpers ----------

def parse_participant_from_name(name: str):
    nums = re.findall(r"\d+", str(name))
    return nums[0] if nums else None

def parse_session_from_name(name: str, first_key="1st", last_key="last") -> str:
    s = str(name).lower()
    if first_key.lower() in s:
        return "First"
    if last_key.lower() in s:
        return "Last"
    return "Unknown"

def ensure_participant_session(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Ensure columns Participant and Session exist; if missing, derive from CSV_name.
    Returns (df_out, derivation_note).
    """
    out = df.copy()
    colmap = {c.lower(): c for c in out.columns}

    # Participant
    if "participant" in colmap:
        out["Participant"] = out[colmap["participant"]].astype(str).str.strip()
        part_src = "Participant"
    elif "csv_name" in colmap:
        out["Participant"] = out[colmap["csv_name"]].apply(parse_participant_from_name)
        part_src = "CSV_name"
    else:
        raise ValueError("Need either 'Participant' or 'CSV_name' to infer participants.")

    # Session
    if "session" in colmap:
        out["Session"] = out[colmap["session"]].astype(str).str.strip().str.capitalize()
        sess_src = "Session"
    elif "csv_name" in colmap:
        out["Session"] = out[colmap["csv_name"]].apply(parse_session_from_name)
        sess_src = "CSV_name"
    else:
        raise ValueError("Need either 'Session' or 'CSV_name' to infer sessions.")

    # Keep First/Last only
    keep = out["Session"].isin(["First", "Last"])
    dropped = int((~keep).sum())
    if dropped:
        print(f"ℹ️  Dropping {dropped} rows with Session not in {{First, Last}}.")
    out = out.loc[keep].copy()

    return out, f"Participant:{part_src}; Session:{sess_src}"

def split_patterns(spec: str) -> List[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]

def glob_to_regex(glob: str) -> str:
    return "^" + re.escape(glob).replace(r"\*", ".*") + "$"


# ---------- plotting ----------

def plot_delta_heatmap(delta_df: pd.DataFrame, title: str, out_png: Path):
    """
    delta_df: index=Participant, columns=Delta_<feature>, values=Last-First
    """
    if delta_df.empty:
        return
    # sort columns for readability
    cols_sorted = sorted(delta_df.columns)
    Z = delta_df[cols_sorted].to_numpy(dtype=float)
    plt.figure(figsize=(max(8, 0.25*len(cols_sorted)), max(4, 0.3*len(delta_df.index))))
    plt.imshow(Z, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Δ (Last - First)")
    plt.yticks(range(len(delta_df.index)), delta_df.index)
    plt.xticks(range(len(cols_sorted)), cols_sorted, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ---------- core ----------

def compute_family_delta(df: pd.DataFrame, pattern_glob: str) -> pd.DataFrame:
    """
    Returns a wide DataFrame indexed by Participant with Delta_<feature> columns.
    """
    regex = glob_to_regex(pattern_glob) if not pattern_glob.startswith("regex:") else pattern_glob.split("regex:",1)[1]
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and re.match(regex, c)]
    if not cols:
        return pd.DataFrame()
    wide = (df[["Participant","Session"] + cols]
            .pivot(index="Participant", columns="Session", values=cols))
    # handle missing First/Last for some participants
    missing_sides = [lev for lev in ["First","Last"] if lev not in wide.columns.levels[1]]
    if missing_sides:
        # create empty side(s) with NaN to allow subtraction
        for ms in missing_sides:
            for c in cols:
                wide[(c, ms)] = np.nan
        # ensure consistent column order
        wide = wide.reindex(columns=pd.MultiIndex.from_product([cols, ["First","Last"]]))

    last = wide.xs("Last", level=1, axis=1)
    first = wide.xs("First", level=1, axis=1)
    d = (last - first)
    d.columns = [f"Delta_{c}" for c in d.columns]
    d.index.name = "Participant"
    return d


def main():
    ap = argparse.ArgumentParser(description="Compute First→Last deltas for feature families.")
    ap.add_argument("--unified", type=Path, default=Path("unified_features.csv"),
                    help="Unified features CSV from runner.py")
    ap.add_argument("--outdir", type=Path, default=Path("mdi_out"), help="Output directory")
    ap.add_argument("--include", type=str, default="*_STD,*_CV,*_ZeroCross",
                    help="Comma-separated patterns (glob or regex:...) for families to delta.")
    ap.add_argument("--plot", action="store_true", help="Save heatmaps for each delta family.")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.unified)
    df, derivation_note = ensure_participant_session(df_raw)

    patterns = split_patterns(args.include)
    manifest_rows = []

    for pat in patterns:
        delta_wide = compute_family_delta(df, pat)
        tag = pat.replace("*", "").replace("regex:", "REGEX_")
        # Standardize common tags for the three canonical families
        if pat.strip() == "*_STD": tag_clean = "STD"
        elif pat.strip() == "*_CV": tag_clean = "CV"
        elif pat.strip() == "*_ZeroCross": tag_clean = "ZeroCross"
        else: tag_clean = tag if tag else "CUSTOM"

        out_csv = outdir / f"deltas_{tag_clean}.csv"
        delta_wide.reset_index().to_csv(out_csv, index=False)
        print(f"✅ Saved: {out_csv}")

        if args.plot:
            png = outdir / f"heatmap_deltas_{tag_clean}.png"
            plot_delta_heatmap(delta_wide, f"Δ (Last-First) — {tag_clean}", png)
            print(f"🖼️  Saved: {png}")

        manifest_rows.append({
            "family": tag_clean,
            "pattern": pat,
            "n_participants": int(delta_wide.shape[0]),
            "n_features": int(delta_wide.shape[1]),
            "participant_session_source": derivation_note
        })

    # Write a small manifest of choices
    pd.DataFrame(manifest_rows).to_csv(outdir / "compute_deltas_manifest.csv", index=False)
    print("📝 Wrote manifest:", outdir / "compute_deltas_manifest.csv")
    print("Done. All Δ-tables saved in", outdir.resolve())


if __name__ == "__main__":
    main()
