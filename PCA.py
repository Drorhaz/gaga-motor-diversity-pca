#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA / MDI builder (interactive or headless)

Adds:
- Global correlation (pre-pruning): *_STD,*_CV,*_ZeroCross,*_RMS + any columns starting with 'Mean'
- Subset correlation (used for pruning): only the PCA subset (*_STD,*_CV,*_ZeroCross)
- Arrows plot: First=red dot, Last=blue dot, per-participant colored line + legends

Usage (headless):
  python PCA.py --in unified_features.csv --outdir mdi_out \
    --include "*_STD,*_CV,*_ZeroCross" --max-missing 0.20 --corr 0.95 \
    --autoprune --keep "CV,ZeroCross,STD" --k 3 --plot-pcs "1,2" --noninteractive
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# ============== Helpers ==============

def prompt_path(prompt_text: str, must_exist: bool = True, default: Optional[str] = None) -> Path:
    while True:
        s = input(f"{prompt_text}{' ['+default+']' if default else ''}: ").strip()
        if not s and default:
            s = default
        p = Path(s)
        if not must_exist or p.exists():
            return p
        print("❌ Path does not exist. Try again.")

def prompt_float(prompt_text: str, default: float) -> float:
    while True:
        s = input(f"{prompt_text} [{default}]: ").strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            print("❌ Please enter a number.")

def prompt_yesno(prompt_text: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    while True:
        s = input(f"{prompt_text} ({default}): ").strip().lower()
        if not s:
            return default_yes
        if s in ("y", "yes"): return True
        if s in ("n", "no"): return False
        print("❌ Please answer y/n.")

def parse_participant_from_name(name: str) -> Optional[str]:
    nums = re.findall(r"\d+", str(name))
    return nums[0] if nums else None

def parse_session_from_name(name: str, first_key="1st", last_key="last") -> str:
    s = str(name).lower()
    if first_key.lower() in s:
        return "First"
    if last_key.lower() in s:
        return "Last"
    return "Unknown"

def ensure_participant_session(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    colmap = {c.lower(): c for c in out.columns}

    part_col = colmap.get("participant")
    sess_col = colmap.get("session")
    csv_name_col = colmap.get("csv_name")

    if part_col is None:
        if csv_name_col is None:
            raise ValueError("Neither 'Participant' nor 'CSV_name' found to infer participant IDs.")
        out["Participant"] = out[csv_name_col].apply(parse_participant_from_name)
    else:
        out["Participant"] = out[part_col].astype(str).str.strip()

    if sess_col is None:
        if csv_name_col is None:
            raise ValueError("Neither 'Session' nor 'CSV_name' found to infer sessions.")
        out["Session"] = out[csv_name_col].apply(parse_session_from_name)
    else:
        out["Session"] = out[sess_col].astype(str).str.strip().str.capitalize()

    keep_mask = out["Session"].isin(["First", "Last"])
    dropped = (~keep_mask).sum()
    if dropped:
        print(f"ℹ️  Dropping {dropped} rows with Session not in {{First, Last}}.")
    return out.loc[keep_mask].copy()

def select_feature_columns(df: pd.DataFrame, include_patterns: List[str]) -> List[str]:
    cols = df.columns.tolist()
    sel = set()
    for pat in include_patterns:
        pat = pat.strip()
        if not pat:
            continue
        if pat.startswith("regex:"):
            rx = re.compile(pat.split("regex:", 1)[1])
            sel.update([c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and rx.search(c)])
        else:
            # glob-like
            pat_rx = re.compile("^" + re.escape(pat).replace("\\*", ".*") + "$")
            sel.update([c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and pat_rx.match(c)])
    return sorted(sel)

def drop_toomany_nans(df: pd.DataFrame, max_missing: float) -> Tuple[pd.DataFrame, List[str]]:
    frac = df.isna().mean()
    to_drop = frac[frac > max_missing].index.tolist()
    return df.drop(columns=to_drop), to_drop

def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))

def zscore(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    return pd.DataFrame(X, index=df.index, columns=df.columns), scaler

def find_high_corr_pairs(corr: pd.DataFrame, thresh: float):
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= thresh:
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs

def greedy_auto_prune(pairs, keep_preference: List[str]) -> List[str]:
    dropped = set()
    def score(name: str) -> int:
        # higher score => stronger keep
        return sum(int(k.lower() in name.lower()) for k in keep_preference)
    for a, b, r in pairs:
        if a in dropped or b in dropped:
            continue
        sa, sb = score(a), score(b)
        if sa > sb:
            dropped.add(b)
        elif sb > sa:
            dropped.add(a)
        else:
            dropped.add(sorted([a, b])[1])  # deterministic tie-break
    return sorted(dropped)

def plot_scree(explained, outpath: Path):
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(explained)*100, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_corr_heatmap(corr: pd.DataFrame, out_png: Path, title: str):
    plt.figure(figsize=(max(6, 0.25*len(corr.columns)), max(5, 0.25*len(corr.columns))))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", square=True, cbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_arrows(scores_df: pd.DataFrame, pcx: int, pcy: int, outpath: Path):
    plt.figure(figsize=(9,7))
    ax = plt.gca()

    # participant color map for lines
    cmap = plt.get_cmap("tab20")
    pids = sorted(scores_df["Participant"].unique())
    color_map = {pid: cmap(i % cmap.N) for i, pid in enumerate(pids)}

    # legend proxies for First/Last
    first_proxy = Line2D([0], [0], marker='o', color='w', label='First', markerfacecolor='red', markersize=6)
    last_proxy  = Line2D([0], [0], marker='o', color='w', label='Last',  markerfacecolor='blue', markersize=6)

    # also build line proxies for participants
    part_proxies = [Line2D([0],[0], color=color_map[pid], lw=2, label=str(pid)) for pid in pids]

    for pid, g in scores_df.groupby("Participant"):
        g = g.set_index("Session")
        if {"First","Last"}.issubset(g.index):
            x0,y0 = float(g.loc["First", "PCx"]), float(g.loc["First", "PCy"])
            x1,y1 = float(g.loc["Last",  "PCx"]), float(g.loc["Last",  "PCy"])
            ax.plot([x0,x1],[y0,y1], color=color_map[pid], linewidth=2, alpha=0.9)
            ax.scatter([x0],[y0], c="red",  s=30, zorder=3)
            ax.scatter([x1],[y1], c="blue", s=30, zorder=3)

    ax.set_xlabel(f"PC{pcx}")
    ax.set_ylabel(f"PC{pcy}")
    ax.set_title(f"Participants in PC{pcx}–PC{pcy} Space (First→Last)")
    ax.grid(True, alpha=0.3)

    # two legends: First/Last + participant colors
    leg1 = ax.legend(handles=[first_proxy, last_proxy], loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=part_proxies, title="Participant", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


# ============== Headless args ==============

def build_argparser():
    ap = argparse.ArgumentParser(description="PCA/MDI builder (interactive or headless). Omit flags to be prompted.")
    ap.add_argument("--in", dest="in_path", type=str, help="Unified features CSV.")
    ap.add_argument("--outdir", type=str, default="mdi_out", help="Output directory.")
    ap.add_argument("--include", type=str, default="*_STD,*_CV,*_ZeroCross",
                    help='Comma-separated glob/regex for PCA subset, e.g., "*_STD,*_CV,*_ZeroCross"')
    ap.add_argument("--max-missing", type=float, default=0.20, help="Drop features with NaN fraction > this.")
    ap.add_argument("--corr", type=float, default=0.95, help="Correlation threshold |r| for flagging/pruning.")
    ap.add_argument("--autoprune", action="store_true", help="Greedy auto-prune highly correlated pairs.")
    ap.add_argument("--keep", type=str, default="CV,ZeroCross,STD", help="Preference order for auto-prune.")
    ap.add_argument("--k", type=int, default=3, help="PCs to include in MDI kD (variance-weighted).")
    ap.add_argument("--plot-pcs", type=str, default="1,2", help="Which PCs to plot arrows for, e.g., '1,2'.")
    ap.add_argument("--noninteractive", action="store_true", help="Do not prompt; use defaults/flags.")
    return ap


# ============== Main ==============

def main():
    args = build_argparser().parse_args()

    interactive = not args.noninteractive and (args.in_path is None)

    if interactive:
        print("\n=== Interactive PCA / MDI Builder ===\n")
        in_path = prompt_path("Enter path to unified features CSV", must_exist=True)
        outdir = prompt_path("Enter output directory", must_exist=False, default="mdi_out")
        include_str = input("\nWhich features to include? (glob/regex) [*_STD,*_CV,*_ZeroCross]: ").strip() or "*_STD,*_CV,*_ZeroCross"
        include_patterns = [s.strip() for s in include_str.split(",") if s.strip()]
        max_missing = prompt_float("\nDrop features with fraction NaNs greater than", 0.20)
        corr_thresh = prompt_float("Flag pairs with |r| ≥ (correlation threshold)", 0.95)
        do_autoprune = prompt_yesno("Auto-prune highly correlated features (greedy)?", True)
        keep_pref_str = "CV,ZeroCross,STD"
        if do_autoprune:
            keep_pref_str = input("Preference hints (comma-separated) [CV,ZeroCross,STD]: ").strip() or "CV,ZeroCross,STD"
        k = 3
        pcs_in = input("\nWhich PCs to plot with First→Last arrows? (e.g., '1,2') [1,2]: ").strip() or "1,2"
    else:
        in_path = Path(args.in_path)
        outdir = Path(args.outdir)
        include_patterns = [s.strip() for s in args.include.split(",") if s.strip()]
        max_missing = args.max_missing
        corr_thresh = args.corr
        do_autoprune = args.autoprune
        keep_pref_str = args.keep
        k = max(2, args.k)
        pcs_in = args.plot_pcs

    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    print("\n📂 Loading CSV ...")
    df_raw = pd.read_csv(in_path)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.match(r"^Unnamed")]
    print(f"   Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")

    # Ensure Participant/Session
    print("🧭 Checking Participant & Session columns ...")
    try:
        df = ensure_participant_session(df_raw)
    except Exception as e:
        print("❌", e)
        return
    print(f"   Kept rows (First/Last): {len(df)}")
    print("   Participants:", df['Participant'].nunique())

    # ---------- Define sets ----------
    # PCA subset (user-controlled)
    subset_cols = select_feature_columns(df, include_patterns)
    print(f"🔎 PCA-subset numeric features: {len(subset_cols)}")
    if len(subset_cols) == 0:
        print("❌ No numeric features matched the PCA subset patterns. Exiting.")
        return
    (outdir / "features_initial_selected.txt").write_text("\n".join(subset_cols))

    # Global set (pre-pruning): *_STD,*_CV,*_ZeroCross,*_RMS + any column starting with 'Mean'
    def is_numeric_series(colname: str) -> bool:
        return pd.api.types.is_numeric_dtype(df[colname])

    # Global set (pre-pruning): *_STD,*_CV,*_ZeroCross,*_RMS + the exact MeanActiveJoints column if present
    global_cols = []
    num = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}

    # pattern-based families
    for c in num:
        if re.search(r"(_STD|_CV|_ZeroCross|_RMS)$", str(c)):
            global_cols.append(c)

    # exact extras from features.py
    if "MeanActiveJoints" in num:
        global_cols.append("MeanActiveJoints")

    global_cols = sorted(set(global_cols))
    print(f"🌐 Global correlation feature set: {len(global_cols)}")

    # ---------- Missingness / impute / z-score ----------
    # Subset flow (used for pruning + PCA)
    X_sub = df[subset_cols].copy()
    X_sub, dropped_nan_sub = drop_toomany_nans(X_sub, max_missing)
    if dropped_nan_sub:
        print(f"   (subset) Dropped {len(dropped_nan_sub)} features (> {max_missing:.0%} NaNs).")
        (outdir / "subset_dropped_nan_features.txt").write_text("\n".join(dropped_nan_sub))
    if X_sub.isna().any().any():
        X_sub = median_impute(X_sub)
        print("   (subset) ✅ Median-imputed.")

    Xz_sub, _ = zscore(X_sub)

    # Global flow (pre-pruning diagnostics only)
    if global_cols:
        X_glob = df[global_cols].copy()
        X_glob, dropped_nan_glob = drop_toomany_nans(X_glob, max_missing)
        if dropped_nan_glob:
            print(f"   (global) Dropped {len(dropped_nan_glob)} features (> {max_missing:.0%} NaNs) from global corr.")
            (outdir / "global_dropped_nan_features.txt").write_text("\n".join(dropped_nan_glob))
        if X_glob.isna().any().any():
            X_glob = median_impute(X_glob)
            print("   (global) ✅ Median-imputed.")
        Xz_glob, _ = zscore(X_glob)
    else:
        Xz_glob = None

    # ---------- Correlation diagnostics ----------
    # Global correlation
    if Xz_glob is not None and Xz_glob.shape[1] >= 2:
        print("\n🔗 Global correlation (pre-pruning) ...")
        corr_glob = Xz_glob.corr()
        corr_glob.to_csv(outdir / "corr_global.csv")
        plot_corr_heatmap(corr_glob, outdir / "corr_global.png",
                          "Global Correlation: *_STD,*_CV,*_ZeroCross,*_RMS + Mean*")
        # high-|r| pairs for global
        pairs_glob = find_high_corr_pairs(corr_glob, corr_thresh)
        pd.DataFrame(pairs_glob, columns=["feat_a","feat_b","r"]).to_csv(outdir / "high_corr_pairs_global.csv", index=False)
        print(f"   Saved global correlation → {outdir/'corr_global.csv'} / corr_global.png")
        if pairs_glob:
            print(f"   Global high-|r| pairs (|r| ≥ {corr_thresh}): {len(pairs_glob)} (see high_corr_pairs_global.csv)")
    else:
        print("\n🔗 Global correlation: skipped (not enough global features).")

    # Subset correlation (used for pruning)
    print("\n🔗 Subset correlation (used for pruning) ...")
    corr_sub = Xz_sub.corr()
    corr_sub.to_csv(outdir / "corr_subset.csv")
    plot_corr_heatmap(corr_sub, outdir / "corr_subset.png",
                      "Subset Correlation: PCA features")
    print(f"   Saved subset correlation → {outdir/'corr_subset.csv'} / corr_subset.png")

    # High-|r| pairs on subset
    pairs = find_high_corr_pairs(corr_sub, corr_thresh)
    pd.DataFrame(pairs, columns=["feat_a","feat_b","r"]).to_csv(outdir / "high_corr_pairs.csv", index=False)
    if pairs:
        print(f"\n⚠️  Highly correlated pairs in subset (|r| ≥ {corr_thresh}): top {min(20, len(pairs))}")
        for a, b, r in pairs[:20]:
            print(f"   {a:40s}  ~  {b:40s}   r={r:+.3f}")
    else:
        print(f"   No subset pairs at |r| ≥ {corr_thresh}")

    # Pruning (on subset only)
    features_to_use = Xz_sub.columns.tolist()
    if pairs:
        if do_autoprune:
            keep_pref = [s.strip() for s in keep_pref_str.split(",") if s.strip()]
            drops = greedy_auto_prune(pairs, keep_pref)
            features_to_use = [f for f in features_to_use if f not in drops]
            (outdir / "auto_pruned_features.txt").write_text("\n".join(drops))
            print(f"   ✂️  Auto-pruned {len(drops)} subset features.")
        elif interactive:
            manual = input("Enter comma-separated features to drop (or leave empty): ").strip()
            if manual:
                drop_list = [s.strip() for s in manual.split(",") if s.strip()]
                features_to_use = [f for f in features_to_use if f not in drop_list]
                (outdir / "manually_dropped_features.txt").write_text("\n".join(drop_list))
                print(f"   ✂️  Manually dropped {len(drop_list)} subset features.")

    (outdir / "final_features_used.txt").write_text("\n".join(features_to_use))
    X_final = Xz_sub[features_to_use]
    print(f"✅ Final feature count (subset for PCA): {X_final.shape[1]}")

    # ---------- PCA ----------
    print("\n🧮 Running PCA ...")
    pca = PCA()
    scores = pca.fit_transform(X_final.values)
    expl = pca.explained_variance_ratio_
    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(expl))],
        "ExplainedVar": expl,
        "CumulativeVar": np.cumsum(expl)
    })
    var_df.to_csv(outdir / "pca_variance_explained.csv", index=False)
    print("   Explained variance (first 10 PCs):")
    print(var_df.head(10).to_string(index=False))

    plot_scree(expl, outdir / "pca_scree.png")
    print(f"   Saved scree plot → {outdir/'pca_scree.png'}")

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_final.columns,
        columns=[f"PC{i+1}" for i in range(len(expl))]
    )
    loadings.to_csv(outdir / "pca_loadings.csv")
    print(f"   Saved all loadings → {outdir/'pca_loadings.csv'}")

    # Long-form loadings + family tag
    load_long = (loadings.abs()
                 .rename_axis("feature")
                 .reset_index()
                 .melt(id_vars="feature", var_name="PC", value_name="abs_loading"))
    load_long["family"] = np.where(load_long["feature"].str.endswith("_STD"), "STD",
                          np.where(load_long["feature"].str.endswith("_CV"), "CV",
                          np.where(load_long["feature"].str.endswith("_ZeroCross"), "ZeroCross",
                          np.where(load_long["feature"].str.endswith("_RMS"), "RMS","Other"))))
    load_long.to_csv(outdir / "pca_loadings_long.csv", index=False)

    # Barplots for PC1..PC3
    for pc in [c for c in loadings.columns[:3]]:
        top = (load_long[load_long["PC"]==pc]
               .sort_values("abs_loading", ascending=False).head(12))
        plt.figure(figsize=(9,5))
        plt.barh(top["feature"][::-1], top["abs_loading"][::-1])
        plt.xlabel("|loading|"); plt.title(f"Top loadings for {pc}")
        plt.tight_layout()
        plt.savefig(outdir / f"loadings_{pc}.png", dpi=160); plt.close()

    # Interpretation table
    K = 20
    interp_rows = []
    for i, pc in enumerate(loadings.columns[:3], start=1):
        topK = (load_long[load_long["PC"]==pc]
                .sort_values("abs_loading", ascending=False)
                .head(K))
        fam_sum = topK.groupby("family")["abs_loading"].sum().sort_values(ascending=False)
        dominant_family = fam_sum.index[0] if not fam_sum.empty else "NA"
        label = {"STD":"overall variability","CV":"relative variability",
                 "ZeroCross":"oscillatory variability","RMS":"amplitude"}\
                .get(dominant_family, "mixed")
        interp_rows.append({
            "PC": f"PC{i}",
            "DominantFamily": dominant_family,
            "ProposedLabel": label,
            "ExplainedVar%": expl[i-1]*100.0,
            "TopFeatures": "; ".join(topK["feature"].tolist())
        })
    pd.DataFrame(interp_rows).to_csv(outdir / "pc_interpretation.csv", index=False)
    print(f"   Saved PC interpretation → {outdir/'pc_interpretation.csv'}")

    # Print top contributors for PC1 & PC2
    if len(expl) >= 2:
        top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
        top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(10)
        print("\n🔎 Top 10 |loadings| for PC1:")
        print(top_pc1.to_string())
        print("\n🔎 Top 10 |loadings| for PC2:")
        print(top_pc2.to_string())

    # Scores with metadata
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(len(expl))])
    scores_df["Participant"] = df["Participant"].values
    scores_df["Session"] = df["Session"].values
    scores_df.to_csv(outdir / "pca_scores.csv", index=False)
    print(f"\n   Saved PCA scores → {outdir/'pca_scores.csv'}")

    # kD MDI (variance-weighted)
    k = max(2, min(k, len(expl)))
    use_cols = [f"PC{i+1}" for i in range(k)]
    scores_k = scores_df[["Participant","Session"] + use_cols].copy()

    first_k = scores_k[scores_k["Session"]=="First"].set_index("Participant")[use_cols]
    last_k  = scores_k[scores_k["Session"]=="Last" ].set_index("Participant")[use_cols]
    common = sorted(set(first_k.index) & set(last_k.index))

    rows = []
    w = expl[:k] / np.sum(expl[:k]) if np.sum(expl[:k]) > 0 else np.ones(k)/k
    for pid in common:
        f = first_k.loc[pid].values.astype(float)
        l = last_k.loc[pid].values.astype(float)
        d = l - f
        deuc = float(np.linalg.norm(d))
        dweuc = float(np.sqrt(np.sum(w * d**2)))
        row = {"Participant": pid, "MDI_Euclid_kD": deuc, "MDI_VarWeighted_kD": dweuc}
        for i in range(k):
            row[f"Delta_PC{i+1}"] = float(d[i])
        rows.append(row)

    mdi_k = pd.DataFrame(rows)
    mdi_k.to_csv(outdir / f"mdi_deltas_PC1..PC{k}.csv", index=False)
    print(f"   Saved kD MDI table → {outdir / f'mdi_deltas_PC1..PC{k}.csv'}")

    # Arrows plot + 2D delta table
    try:
        pcx, pcy = [int(s) for s in pcs_in.split(",")]
    except Exception:
        pcx, pcy = 1, 2
    pcx = max(1, min(pcx, len(expl))); pcy = max(1, min(pcy, len(expl)))
    print(f"\n🖼️  Plotting PC{pcx} vs PC{pcy} ...")

    sdf = scores_df.rename(columns={f"PC{pcx}":"PCx", f"PC{pcy}":"PCy"})
    plot_arrows(sdf, pcx, pcy, outdir / f"pc_arrows_PC{pcx}_PC{pcy}.png")
    print(f"   Saved arrows plot → {outdir / f'pc_arrows_PC{pcx}_PC{pcy}.png'}")

    first = sdf[sdf["Session"]=="First"].set_index("Participant")[["PCx","PCy"]]
    last  = sdf[sdf["Session"]=="Last" ].set_index("Participant")[["PCx","PCy"]]
    common2 = sorted(set(first.index) & set(last.index))
    if not common2:
        print("⚠️  No participants with both First and Last. Skipping 2D Δ table.")
    else:
        rows2 = []
        for pid in common2:
            f = first.loc[pid]; l = last.loc[pid]
            dpcx = float(l["PCx"] - f["PCx"])
            dpcy = float(l["PCy"] - f["PCy"])
            deuc2 = float(np.linalg.norm(l.values - f.values))
            rows2.append({"Participant": pid, f"Delta_PC{pcx}": dpcx, f"Delta_PC{pcy}": dpcy,
                          f"Delta_Euclid_PC{pcx}_{pcy}": deuc2})
        pd.DataFrame(rows2).to_csv(outdir / f"mdi_deltas_PC{pcx}_{pcy}.csv", index=False)
        print(f"   Saved 2D MDI table → {outdir / f'mdi_deltas_PC{pcx}_{pcy}.csv'}")

    print("\n✅ Done. All outputs saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
