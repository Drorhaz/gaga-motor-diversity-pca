#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py — Viewpoint + size normalization and low-pass filtering (vectorized, NaN-safe)

Pipeline (per CSV):
  1) Viewpoint normalize coordinates to a body-fixed frame (hips/spine axes).
  2) Size normalize by shoulder width (meters).
  3) Low-pass filter each numeric *_Position_{X,Y,Z} column (Butterworth).
  4) Preserve 'Time' and any non-position columns unchanged.

Usage
-----
From cleaned data (after check_sampling.py):
  python preprocess.py --in data_clean/sample_uniform_120Hz.csv \
      --outdir data_preprocessed \
      --participants participants_shoulder_width.xlsx \
      --participant-col Participant --shoulder-col shoulder_width_m \
      --fps 120 --cutoff 8 --order 4

If you don’t have a participants file, pass a fixed width:
  python preprocess.py --in data_clean/sample_uniform_120Hz.csv \
      --outdir data_preprocessed --shoulder 0.39 --fps 120

Notes
-----
- Viewpoint basis per frame:
   x̂ = normalize(LeftUpLeg − RightUpLeg)
   ŷ = normalize(Spine1 − 0.5*(LeftUpLeg+RightUpLeg)), then orthogonalize via ŷ := normalize(ẑ×x̂)
   ẑ = normalize(x̂×ŷ)
- Degenerate frames (zero-length vectors) reuse the previous valid basis; if none exists yet, identity is used.
- Filtering is NaN-safe: short gaps are linearly interpolated before filtfilt and NaNs restored after.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ======= Defaults / constants =======
AXES = ["X", "Y", "Z"]
POS_SUFFIX = "_Position_"
DEFAULT_FPS = 120.0
DEFAULT_CUTOFF = 8.0
DEFAULT_ORDER = 4

# Joints used for the body frame
HIP_CENTER = "Hips"
HIP_L = "LeftUpLeg"
HIP_R = "RightUpLeg"
SPINE = "Spine1"

# ---------- Column helpers ----------
def position_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns
            if isinstance(c, str)
            and POS_SUFFIX in c
            and c.rsplit(POS_SUFFIX, 1)[-1] in AXES
            and pd.api.types.is_numeric_dtype(df[c])]

def joint_triplet(df: pd.DataFrame, joint: str) -> List[str]:
    return [f"{joint}{POS_SUFFIX}{a}" for a in AXES if f"{joint}{POS_SUFFIX}{a}" in df.columns]

def parse_participant_from_stem(stem: str) -> Optional[str]:
    m = re.findall(r"\d+", stem)
    return m[0] if m else None

# ---------- Linear algebra utilities (vectorized over frames N) ----------
def _row_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # v: (N,3) -> norms: (N,1)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < eps] = np.nan  # mark degenerate
    return n

def _normalize_rows(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _row_norm(v)
    out = v / n
    return out, np.isfinite(n.squeeze())  # unit vectors, valid mask

def _carry_forward_rows(M: np.ndarray, valid_mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """
    Carry forward last valid row of M where valid_mask is False.
    If no prior valid row exists, use fallback (1xK).
    """
    M = M.copy()
    last = fallback.reshape(1, -1)
    for i in range(M.shape[0]):
        if valid_mask[i]:
            last = M[i:i+1]
        else:
            M[i] = last
    return M

def build_body_axes(Lhip: np.ndarray, Rhip: np.ndarray, Spine: np.ndarray) -> np.ndarray:
    """
    Given joint arrays (N,3) for LeftUpLeg, RightUpLeg, Spine1, return per-frame
    basis matrix R with shape (N,3,3) whose rows are [x̂; ŷ; ẑ].
    """
    # x̂: left-right axis
    x_raw = Lhip - Rhip
    x_hat, x_valid = _normalize_rows(x_raw)

    # ŷ: vertical-ish (spine - midhip)
    mid_hip = 0.5 * (Lhip + Rhip)
    y_raw = Spine - mid_hip
    y_hat, y_valid = _normalize_rows(y_raw)

    # ẑ: orthogonal out-of-plane via cross(x̂, ŷ)
    # For any invalid rows, we'll fill later
    z_raw = np.cross(x_hat, y_hat)
    z_hat, z_valid = _normalize_rows(z_raw)

    # Re-orthogonalize ŷ := normalize(ẑ × x̂)
    y2_raw = np.cross(z_hat, x_hat)
    y2_hat, y2_valid = _normalize_rows(y2_raw)

    # Validity: need x̂ and ẑ valid to define y2̂
    valid = x_valid & z_valid & y2_valid

    # Carry-forward for degenerate frames
    # Fallback axes = identity rows
    I = np.eye(3)
    x_hat = _carry_forward_rows(x_hat, x_valid, I[0])
    z_hat = _carry_forward_rows(z_hat, z_valid, I[2])
    y2_hat = _carry_forward_rows(y2_hat, y2_valid, I[1])

    # Stack rows to R (N,3,3)
    R = np.stack([x_hat, y2_hat, z_hat], axis=1)
    return R  # rows = basis vectors

def rotate_positions(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Project joint vectors P (N,3) into per-frame basis R (N,3,3).
    Result: P' (N,3) where each row is [dot(P,x̂), dot(P,ŷ), dot(P,ẑ)].
    """
    return np.einsum("nij,nj->ni", R, P)

# ---------- NaN-safe filtering ----------
def butterworth_filter_series(x: np.ndarray, fps: float, cutoff: float, order: int) -> np.ndarray:
    """
    NaN-safe filtfilt:
      - Temporarily linearly interpolate short NaN gaps (endpoints: ffill/bfill).
      - Skip filtfilt if signal too short for padding; return as-is.
      - Restore NaNs to original positions after filtering.
    """
    x = x.astype(float)
    n = len(x)

    # If too short for filtfilt padding, return as-is
    from math import ceil
    b, a = butter(order, cutoff / (0.5 * fps), btype='low')
    padlen = 3 * (max(len(a), len(b)) - 1)
    if n <= padlen + 1:
        return x.copy()

    # Record NaNs
    nan_mask = ~np.isfinite(x)
    if nan_mask.all():
        return x.copy()

    # Interpolate over NaNs for filtering only
    xi = x.copy()
    idx = np.arange(n)
    if nan_mask.any():
        # forward/back fill edges
        first_valid = np.argmax(~nan_mask)
        last_valid = n - 1 - np.argmax((~nan_mask)[::-1])
        xi[:first_valid] = x[first_valid]
        xi[last_valid+1:] = x[last_valid]
        # linear interp for interior NaNs
        good = ~nan_mask
        xi[nan_mask] = np.interp(idx[nan_mask], idx[good], xi[good])

    # Filter
    yf = filtfilt(b, a, xi, axis=0, padlen=padlen)

    # Restore NaNs
    yf[nan_mask] = np.nan
    return yf

# ---------- Core preprocessing ----------
def preprocess_motion_data(in_csv: Path,
                           outdir: Path,
                           shoulder_width_m: float,
                           fps: float = DEFAULT_FPS,
                           cutoff: float = DEFAULT_CUTOFF,
                           order: int = DEFAULT_ORDER) -> Path:
    """
    Apply viewpoint + size normalization and low-pass filtering to all *_Position_* columns.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_csv)

    # Keep Time column (if present) and any non-position columns
    time_col = None
    for c in df.columns:
        if str(c).lower() == "time":
            time_col = c
            break

    pos_cols = position_columns(df)
    if not pos_cols:
        raise ValueError("No numeric position columns found (*_Position_{X,Y,Z}).")

    # Build joint arrays we need for the basis
    def get_xyz(joint: str) -> np.ndarray:
        cols = joint_triplet(df, joint)
        if len(cols) != 3:
            raise ValueError(f"Missing axes for joint '{joint}'; found {cols}")
        return df[cols].to_numpy(dtype=float)

    # Hip reference (translation)
    H = get_xyz(HIP_CENTER)  # (N,3)
    L = get_xyz(HIP_L)
    R = get_xyz(HIP_R)
    S = get_xyz(SPINE)

    # Build per-frame basis (N,3,3)
    Rbasis = build_body_axes(L, R, S)

    # Prepare output DataFrame
    out = pd.DataFrame(index=df.index)
    if time_col is not None:
        out[time_col] = df[time_col]

    # Apply viewpoint normalization: for every joint, (P - Hips) projected to basis
    # Collect joints present from columns
    joints = sorted({c.split(POS_SUFFIX)[0] for c in pos_cols})
    for j in joints:
        cols = joint_triplet(df, j)
        P = df[cols].to_numpy(dtype=float) - H  # translate to hips
        Prot = rotate_positions(Rbasis, P)      # rotate into body frame
        # Size normalize by shoulder width (meters)
        Prot /= float(shoulder_width_m)
        # Store temporarily; filtering next
        out[cols[0]] = Prot[:, 0]
        out[cols[1]] = Prot[:, 1]
        out[cols[2]] = Prot[:, 2]

    # Low-pass filter each numeric position column (NaN-safe)
    for c in out.columns:
        if c == time_col:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = butterworth_filter_series(out[c].to_numpy(dtype=float),
                                               fps=fps, cutoff=cutoff, order=order)

    # Also carry forward any non-position, non-time columns unchanged
    for c in df.columns:
        if c not in out.columns and not (POS_SUFFIX in str(c) and str(c).rsplit(POS_SUFFIX, 1)[-1] in AXES):
            out[c] = df[c]

    # Save
    stem = in_csv.stem
    out_path = outdir / f"{stem}_preprocessed.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Preprocessed data saved → {out_path}")
    return out_path

# ---------- Participants lookup ----------
def load_shoulder_width(participants_path: Optional[Path],
                        participant_col: str,
                        shoulder_col: str,
                        csv_stem: str,
                        override: Optional[float]) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--shoulder must be > 0.")
        return float(override)

    pid = parse_participant_from_stem(csv_stem)
    if participants_path is None:
        raise ValueError("No --shoulder provided and no --participants file to look up. Please supply one.")
    if not participants_path.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_path}")

    if participants_path.suffix.lower() == ".csv":
        ptab = pd.read_csv(participants_path, dtype=str)
        ptab.columns = ptab.columns.str.strip()  # ✅ normalize headers
    else:
        read_kwargs = {"dtype": str}

    colmap = {c.lower(): c for c in ptab.columns}
    if participant_col.lower() not in colmap or shoulder_col.lower() not in colmap:
        raise ValueError(f"Participants table must include '{participant_col}' and '{shoulder_col}' columns.")
    Pcol = colmap[participant_col.lower()]
    Scol = colmap[shoulder_col.lower()]

    row = ptab.loc[ptab[Pcol].astype(str) == str(pid)]
    if row.empty:
        raise ValueError(f"Participant '{pid}' not found in {participants_path}.")
    width = float(row.iloc[0][Scol])
    if width <= 0:
        raise ValueError(f"Shoulder width must be >0. Got {width}.")
    return width

# ---------- CLI ----------
def build_argparser():
    ap = argparse.ArgumentParser(description="Viewpoint + size normalization and low-pass filtering (NaN-safe).")
    ap.add_argument("--in", dest="in_csv", type=Path, required=True, help="Input CSV (from data_clean).")
    ap.add_argument("--outdir", type=Path, default=Path("data_preprocessed"), help="Output directory.")
    # Shoulder width: either fixed scalar or lookup table
    ap.add_argument("--shoulder", type=float, default=None, help="Override shoulder width in meters (scalar).")
    ap.add_argument("--participants", type=Path, default=None, help="Participants table (csv/xlsx) with shoulder width.")
    ap.add_argument("--participant-col", type=str, default="Participant", help="Column name for participant ID.")
    ap.add_argument("--shoulder-col", type=str, default="shoulder_width_m", help="Column name with shoulder width (meters).")
    # Filtering parameters
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Sampling rate (Hz) for filtering.")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF, help="Butterworth cutoff (Hz).")
    ap.add_argument("--order", type=int, default=DEFAULT_ORDER, help="Butterworth order.")
    return ap

def main():
    args = build_argparser().parse_args()

    width = load_shoulder_width(
        participants_path=args.participants,
        participant_col=args.participant_col,
        shoulder_col=args.shoulder_col,
        csv_stem=args.in_csv.stem,
        override=args.shoulder
    )

    preprocess_motion_data(
        in_csv=args.in_csv,
        outdir=args.outdir,
        shoulder_width_m=width,
        fps=args.fps,
        cutoff=args.cutoff,
        order=args.order
    )

if __name__ == "__main__":
    main()
