#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOPO logistic regression on PCA scores + MeanActiveJoints,
and plotting ΔMeanActiveJoints per participant.

Parallels the structure of lopo_logistic.py:
- Auto-k from MDI (optional)
- LOPO with per-fold z-scoring
- Saves predictions, summary, permutation AUROCs
- Adds: merges MeanActiveJoints per session; computes/saves/plots ΔMeanActiveJoints

Usage
-----
Auto-k and defaults:
  python lopo_logistic_plus_meanactive.py \
    --scores mdi_out/pca_scores.csv \
    --unified mdi_out/unified_features.csv \
    --mdi mdi_out/mdi_deltas_PC1..PC3.csv \
    --outdir mdi_out \
    --permutations 1000 --seed 42

Manual k:
  python lopo_logistic_plus_meanactive.py \
    --scores mdi_out/pca_scores.csv \
    --unified mdi_out/unified_features.csv \
    --k 3 --outdir mdi_out \
    --permutations 1000 --seed 42
"""

import argparse
from pathlib import Path
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, average_precision_score
)

# ------------------------------
# Helpers (kept parallel to lopo_logistic.py)
# ------------------------------

def infer_k_from_mdi(mdi_path: Path) -> Optional[int]:
    try:
        df = pd.read_csv(mdi_path, nrows=1)
    except Exception:
        return None
    pcs = []
    for c in df.columns:
        m = re.match(r"Delta_PC(\d+)$", c)
        if m:
            pcs.append(int(m.group(1)))
    return int(max(pcs)) if pcs else None


def load_scores(scores_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(scores_path)
    colmap = {c.lower(): c for c in df.columns}
    if "participant" not in colmap or "session" not in colmap:
        raise ValueError("pca_scores.csv must contain 'Participant' and 'Session' columns.")
    pc_cols = sorted([c for c in df.columns if re.fullmatch(r"PC\d+", c)],
                     key=lambda s: int(s[2:]))
    if not pc_cols:
        raise ValueError("No PC columns found (expected 'PC1','PC2',...).")
    df = df.copy()
    df["_Participant"] = df[colmap["participant"]].astype(str)
    df["_Session"] = df[colmap["session"]].astype(str).str.capitalize()
    df = df[df["_Session"].isin(["First", "Last"])].reset_index(drop=True)
    return df, pc_cols


def y_from_session(series: pd.Series) -> np.ndarray:
    return (series == "Last").astype(int).to_numpy()


def lopo_split(df: pd.DataFrame, pid_col: str = "_Participant"):
    for pid, idx in df.groupby(pid_col).indices.items():
        test_idx = np.array(sorted(idx))
        train_idx = np.array(sorted(np.setdiff1d(np.arange(len(df)), test_idx)))
        yield pid, train_idx, test_idx


# ------------------------------
# New: MeanActiveJoints integration
# ------------------------------

def load_mean_active(unified_path: Path) -> pd.DataFrame:
    u = pd.read_csv(unified_path)
    # make robust to capitalization
    cmap = {c.lower(): c for c in u.columns}
    required = ["participant", "session", "meanactivejoints"]
    for r in required:
        if r not in cmap:
            raise ValueError("unified_features.csv must contain 'Participant','Session','MeanActiveJoints'.")
    out = u[[cmap["participant"], cmap["session"], cmap["meanactivejoints"]]].copy()
    out.columns = ["Participant", "Session", "MeanActiveJoints"]
    out["Participant"] = out["Participant"].astype(str)
    out["Session"] = out["Session"].astype(str).str.capitalize()
    out = out[out["Session"].isin(["First", "Last"])].reset_index(drop=True)
    return out


def compute_and_save_delta_meanactive(ua: pd.DataFrame, outdir: Path) -> Path:
    """
    ua columns: Participant, Session, MeanActiveJoints (First/Last rows per participant)
    Saves:
      - delta_MeanActiveJoints.csv  (Participant, delta_MeanActiveJoints)
      - delta_MeanActiveJoints.png  (barplot)
    """
    # Wide pivot: columns 'First','Last'
    wide = ua.pivot_table(index="Participant", columns="Session", values="MeanActiveJoints", aggfunc="mean")
    # Ensure both columns exist (fill NaN if missing)
    for col in ["First", "Last"]:
        if col not in wide.columns:
            wide[col] = np.nan
    delta = wide["Last"] - wide["First"]
    delta_df = delta.rename("delta_MeanActiveJoints").reset_index()

    csv_path = outdir / "delta_MeanActiveJoints.csv"
    delta_df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="#888888", linewidth=1)
    ax.bar(delta_df["Participant"].astype(str), delta_df["delta_MeanActiveJoints"])
    ax.set_title("Δ MeanActiveJoints (Last − First) per participant")
    ax.set_ylabel("Δ MeanActiveJoints")
    ax.set_xlabel("Participant")
    plt.xticks(rotation=0)
    plt.tight_layout()
    png_path = outdir / "delta_MeanActiveJoints.png"
    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    return csv_path


# ------------------------------
# LOPO with PCs + MeanActiveJoints and extra metrics
# ------------------------------

def fit_predict_lopo_with_meanactive(df_scores: pd.DataFrame,
                                     pc_cols: List[str],
                                     ua: pd.DataFrame,
                                     k: int,
                                     seed: int = 42):
    """
    Merge MeanActiveJoints into scores and run LOPO on [PC1..k] + MeanActiveJoints.
    Returns pred_df and metrics dict.
    """
    # Merge MeanActiveJoints into PCA table
    add = ua.rename(columns={"Participant": "_Participant", "Session": "_Session"})
    merged = df_scores.merge(add, on=["_Participant", "_Session"], how="left")

    use_cols = pc_cols[:k] + ["MeanActiveJoints"]

    preds = []
    all_true, all_score, fold_losses, fold_aprc, fold_auc, fold_acc = [], [], [], [], [], []

    for pid, tr_idx, te_idx in lopo_split(merged):
        X_train = merged.loc[tr_idx, use_cols].to_numpy(dtype=float)
        X_test  = merged.loc[te_idx, use_cols].to_numpy(dtype=float)
        y_train = y_from_session(merged.loc[tr_idx, "_Session"])
        y_test  = y_from_session(merged.loc[te_idx, "_Session"])

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_test_z  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        clf.fit(X_train_z, y_train)
        prob = clf.predict_proba(X_test_z)[:, 1]
        y_pred = (prob >= 0.5).astype(int)

        fold_df = pd.DataFrame({
            "Participant": merged.loc[te_idx, "_Participant"].values,
            "Session": merged.loc[te_idx, "_Session"].values,
            "y_true": y_test,
            "y_score": prob,
            "MeanActiveJoints": merged.loc[te_idx, "MeanActiveJoints"].values
        }, index=te_idx).sort_index()

        preds.append(fold_df)
        all_true.append(y_test)
        all_score.append(prob)

        # per-fold metrics
        try:
            fold_auc.append(roc_auc_score(y_test, prob))
        except Exception:
            fold_auc.append(np.nan)
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_losses.append(log_loss(y_test, prob, labels=[0,1]))
        fold_aprc.append(average_precision_score(y_test, prob))

    pred_df = pd.concat(preds).sort_index()
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)

    # aggregate metrics (mean across folds)
    metrics = {
        "Accuracy": float(np.nanmean(fold_acc)),
        "AUROC": float(np.nanmean(fold_auc)),
        "LogLoss": float(np.nanmean(fold_losses)),
        "AUPRC": float(np.nanmean(fold_aprc))
    }

    return pred_df, metrics, merged, use_cols


def perm_test_auc(df_merged: pd.DataFrame, use_cols: List[str],
                  observed_auc: float, n_perms: int, seed: int = 42) -> Tuple[float, np.ndarray]:
    """Permutation test (shuffle training labels within each LOPO fold)."""
    if not np.isfinite(observed_auc):
        return float("nan"), np.array([])

    rng = np.random.RandomState(seed)
    perm_aucs = []

    for _ in range(n_perms):
        all_true, all_score = [], []
        for pid, tr_idx, te_idx in lopo_split(df_merged):
            X_train = df_merged.loc[tr_idx, use_cols].to_numpy(dtype=float)
            X_test  = df_merged.loc[te_idx, use_cols].to_numpy(dtype=float)
            y_train = y_from_session(df_merged.loc[tr_idx, "_Session"]).copy()
            y_test  = y_from_session(df_merged.loc[te_idx, "_Session"]).copy()

            rng.shuffle(y_train)

            scaler = StandardScaler()
            X_train_z = scaler.fit_transform(X_train)
            X_test_z  = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
            clf.fit(X_train_z, y_train)
            prob = clf.predict_proba(X_test_z)[:, 1]

            all_true.append(y_test)
            all_score.append(prob)

        y_true = np.concatenate(all_true)
        y_score = np.concatenate(all_score)
        try:
            perm_auc = roc_auc_score(y_true, y_score)
            perm_aucs.append(perm_auc)
        except Exception:
            continue

    perm_aucs = np.array(perm_aucs, dtype=float)
    if perm_aucs.size == 0:
        return float("nan"), perm_aucs

    count_ge = int(np.sum(perm_aucs >= observed_auc))
    pval = (1.0 + count_ge) / (1.0 + len(perm_aucs))
    return float(pval), perm_aucs


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="LOPO logistic regression on PC scores + MeanActiveJoints; plot ΔMeanActiveJoints.")
    ap.add_argument("--scores", required=True, type=Path, help="Path to pca_scores.csv")
    ap.add_argument("--unified", required=True, type=Path, help="Path to unified_features.csv (must have MeanActiveJoints)")
    ap.add_argument("--mdi", type=Path, help="Path to MDI table (e.g., mdi_out/mdi_deltas_PC1..PC3.csv) to infer k automatically.")
    ap.add_argument("--k", type=int, default=3, help="Number of PCs to use if --mdi is not provided or cannot infer.")
    ap.add_argument("--permutations", type=int, default=5000, help="Number of permutations for p-value.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default = scores' parent).")
    args = ap.parse_args()

    outdir = args.outdir if args.outdir is not None else args.scores.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Load core tables
    df_scores, pc_cols = load_scores(args.scores)
    ua = load_mean_active(args.unified)

    # k (auto from MDI if available)
    k_inferred = infer_k_from_mdi(args.mdi) if args.mdi else None
    k = k_inferred if k_inferred is not None else max(1, min(args.k, len(pc_cols)))
    print(f"📐 PCs used: {', '.join(pc_cols[:k])} (k={k})")

    # ΔMeanActiveJoints table + plot
    delta_csv = compute_and_save_delta_meanactive(ua, outdir)
    print(f"📝 Saved ΔMeanActiveJoints table → {delta_csv}")
    print(f"🖼️  Saved ΔMeanActiveJoints plot → {outdir / 'delta_MeanActiveJoints.png'}")

    # LOPO with PCs + MeanActiveJoints
    pred_df, metrics, merged_df, use_cols = fit_predict_lopo_with_meanactive(
        df_scores, pc_cols, ua, k=k, seed=args.seed
    )

    # Save predictions
    pred_path = outdir / "lopo_predictions_plus.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"📝 Saved per-sample predictions → {pred_path}")

    # Permutation test on AUROC
    print(f"🎲 Permutation test with n={args.permutations}, seed={args.seed} ...")
    pval, perm_aucs = perm_test_auc(merged_df, use_cols, observed_auc=metrics["AUROC"],
                                    n_perms=args.permutations, seed=args.seed)

    perm_path = outdir / "lopo_perm_aucs_plus.csv"
    if perm_aucs.size > 0:
        pd.DataFrame({"perm_auroc": perm_aucs}).to_csv(perm_path, index=False)
        print(f"📈 Saved permutation AUROCs → {perm_path}")
        ci_low, ci_high = np.nanpercentile(perm_aucs, [2.5, 97.5])
        perm_mean = float(np.nanmean(perm_aucs))
        perm_std  = float(np.nanstd(perm_aucs, ddof=1))
    else:
        ci_low = ci_high = perm_mean = perm_std = float("nan")

    # Summary table (focused, as requested)
    summary = pd.DataFrame([{
        "k": int(k),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "accuracy": float(metrics["Accuracy"]),
        "auroc": float(metrics["AUROC"]),
        "logloss": float(metrics["LogLoss"]),
        "auprc": float(metrics["AUPRC"]),
        "perm_p": float(pval),
        "perm_auc_mean": float(perm_mean),
        "perm_auc_std": float(perm_std),
        "perm_auc_ci_low_2p5": float(ci_low),
        "perm_auc_ci_high_97p5": float(ci_high),
    }])

    out_path = outdir / "lopo_logistic_plus_results.csv"
    summary.to_csv(out_path, index=False)
    print(f"✅ Saved summary → {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
