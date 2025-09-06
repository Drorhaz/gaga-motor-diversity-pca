#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-One-Participant-Out (LOPO) logistic regression on PCA scores.

Adds:
- Auto-k from MDI (parse Delta_PC* columns)
- Records k, permutations, seed in results
- Saves full permutation AUROC distribution and reports mean/std/95% CI

Usage
-----
Auto-k from MDI:
  python lopo_logistic.py --scores mdi_out/pca_scores.csv \
    --mdi mdi_out/mdi_deltas_PC1..PC3.csv --outdir mdi_out \
    --permutations 1000 --seed 42

Manual k:
  python lopo_logistic.py --scores mdi_out/pca_scores.csv \
    --k 2 --outdir mdi_out --permutations 1000 --seed 42
"""

import argparse
from pathlib import Path
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def infer_k_from_mdi(mdi_path: Path) -> Optional[int]:
    """Infer k from MDI table by counting Delta_PC* columns."""
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
    # normalize column capitalization
    colmap = {c.lower(): c for c in df.columns}
    if "participant" not in colmap or "session" not in colmap:
        raise ValueError("pca_scores.csv must contain 'Participant' and 'Session' columns.")
    # sort PCs
    pc_cols = sorted([c for c in df.columns if re.fullmatch(r"PC\d+", c)],
                     key=lambda s: int(s[2:]))
    if not pc_cols:
        raise ValueError("No PC columns found (expected 'PC1','PC2',...).")
    df = df.copy()
    df["_Participant"] = df[colmap["participant"]].astype(str)
    df["_Session"] = df[colmap["session"]].astype(str).str.capitalize()
    # filter to First/Last only
    df = df[df["_Session"].isin(["First", "Last"])].reset_index(drop=True)
    return df, pc_cols


def y_from_session(series: pd.Series) -> np.ndarray:
    """Map Session to labels: First -> 0, Last -> 1."""
    return (series == "Last").astype(int).to_numpy()


def lopo_split(df: pd.DataFrame, pid_col: str = "_Participant"):
    """Yield (participant_id, train_idx, test_idx) for each LOPO fold."""
    for pid, idx in df.groupby(pid_col).indices.items():
        test_idx = np.array(sorted(idx))
        train_idx = np.array(sorted(np.setdiff1d(np.arange(len(df)), test_idx)))
        yield pid, train_idx, test_idx


def fit_predict_lopo(df: pd.DataFrame, X_cols, seed: int = 42):
    """Run LOPO with per-fold standardization; return (pred_df, acc, auc)."""
    preds = []
    all_true, all_score = [], []

    for pid, tr_idx, te_idx in lopo_split(df):
        X_train = df.loc[tr_idx, X_cols].to_numpy(dtype=float)
        X_test  = df.loc[te_idx, X_cols].to_numpy(dtype=float)
        y_train = y_from_session(df.loc[tr_idx, "_Session"])
        y_test  = y_from_session(df.loc[te_idx, "_Session"])

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_test_z  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        clf.fit(X_train_z, y_train)
        prob = clf.predict_proba(X_test_z)[:, 1]

        fold_df = pd.DataFrame({
            "Participant": df.loc[te_idx, "_Participant"].values,
            "Session": df.loc[te_idx, "_Session"].values,
            "y_true": y_test,
            "y_score": prob
        }, index=te_idx).sort_index()

        preds.append(fold_df)
        all_true.append(y_test)
        all_score.append(prob)

    pred_df = pd.concat(preds).sort_index()
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)

    # Accuracy at 0.5 threshold
    y_pred = (y_score >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))

    # AUROC (may be NaN if only one class present)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")

    return pred_df, acc, auc


def perm_test_auc(df: pd.DataFrame, X_cols, observed_auc: float,
                  n_perms: int, seed: int = 42) -> Tuple[float, np.ndarray]:
    """
    Permutation test: within each LOPO fold, shuffle training labels,
    refit, and compute AUROC across all folds. Returns (p_value, aucs_array).
    p = (1 + #perm_auc >= obs) / (n_perms + 1)
    """
    if not np.isfinite(observed_auc):
        return float("nan"), np.array([])

    rng = np.random.RandomState(seed)
    perm_aucs = []

    for _ in range(n_perms):
        all_true, all_score = [], []
        for pid, tr_idx, te_idx in lopo_split(df):
            X_train = df.loc[tr_idx, X_cols].to_numpy(dtype=float)
            X_test  = df.loc[te_idx, X_cols].to_numpy(dtype=float)
            y_train = y_from_session(df.loc[tr_idx, "_Session"]).copy()
            y_test  = y_from_session(df.loc[te_idx, "_Session"]).copy()

            rng.shuffle(y_train)  # shuffle labels in training fold

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
            auc_perm = roc_auc_score(y_true, y_score)
            perm_aucs.append(auc_perm)
        except Exception:
            # If AUROC cannot be computed for this permutation, skip it
            continue

    perm_aucs = np.array(perm_aucs, dtype=float)
    if perm_aucs.size == 0:
        return float("nan"), perm_aucs

    count_ge = int(np.sum(perm_aucs >= observed_auc))
    pval = (1.0 + count_ge) / (1.0 + len(perm_aucs))
    return float(pval), perm_aucs


def main():
    ap = argparse.ArgumentParser(description="LOPO logistic regression on PCA scores (auto-k from MDI optional).")
    ap.add_argument("--scores", required=True, type=Path, help="Path to pca_scores.csv")
    ap.add_argument("--mdi", type=Path, help="Path to MDI table (e.g., mdi_deltas_PC1..PC3.csv) to infer k automatically.")
    ap.add_argument("--k", type=int, default=3, help="Number of PCs to use if --mdi is not provided or cannot infer.")
    ap.add_argument("--permutations", type=int, default=5000, help="Number of permutations for p-value.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default = scores' parent).")
    args = ap.parse_args()

    # Resolve outdir
    outdir = args.outdir if args.outdir is not None else args.scores.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Load scores and PC columns
    df, pc_cols = load_scores(args.scores)

    # Determine k
    k_inferred = infer_k_from_mdi(args.mdi) if args.mdi else None
    if k_inferred is not None:
        k = k_inferred
        print(f"🔎 Inferred k={k} from MDI file: {args.mdi.name}")
    else:
        k = max(1, min(args.k, len(pc_cols)))
        print(f"ℹ️  Using k={k} (manual / fallback).")

    use_cols = pc_cols[:k]
    print(f"📐 PCs used: {', '.join(use_cols)}")

    # LOPO fit/predict
    pred_df, acc, auc = fit_predict_lopo(df, use_cols, seed=args.seed)

    # Save predictions
    pred_path = outdir / "lopo_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"📝 Saved per-sample predictions → {pred_path}")

    # Permutation test
    print(f"🎲 Permutation test with n={args.permutations}, seed={args.seed} ...")
    pval, perm_aucs = perm_test_auc(df, use_cols, observed_auc=auc, n_perms=args.permutations, seed=args.seed)

    # Save permutation AUROCs (if any)
    perm_path = outdir / "lopo_perm_aucs.csv"
    if perm_aucs.size > 0:
        pd.DataFrame({"perm_auroc": perm_aucs}).to_csv(perm_path, index=False)
        print(f"📈 Saved permutation AUROCs → {perm_path}")
        ci_low, ci_high = np.nanpercentile(perm_aucs, [2.5, 97.5])
        perm_mean = float(np.nanmean(perm_aucs))
        perm_std  = float(np.nanstd(perm_aucs, ddof=1))
    else:
        ci_low = ci_high = perm_mean = perm_std = float("nan")

    # Save summary (records k, permutations, seed)
    summary = pd.DataFrame([{
        "k": int(k),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "accuracy": float(acc),
        "auroc": float(auc),
        "perm_p": float(pval),
        "perm_auc_mean": perm_mean,
        "perm_auc_std": perm_std,
        "perm_auc_ci_low_2p5": float(ci_low),
        "perm_auc_ci_high_97p5": float(ci_high),
    }])
    out_path = outdir / "lopo_logistic_results.csv"
    summary.to_csv(out_path, index=False)
    print(f"✅ Saved summary → {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
