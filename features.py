#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py — joint-level features from angle & velocity series

Computes per joint (for each Angle triplet):
  - RMS  (on velocity, after masking)
  - STD  (on velocity, after masking)
  - CV   (on angle; uses |mean| in denominator; masked)
  - ZeroCross (on velocity; sign changes, ignoring exact zeros)

Also computes:
  - MeanActiveJoints (mean # joints per frame with |velocity| > threshold)

Masking logic:
  - Ingests NaNs from joint_angles.py (invalid frames already set to NaN).
  - Optionally re-enforces angle range [ANGLE_MIN, ANGLE_MAX].
  - Optionally unmask short velocity spikes (<= SPIKE_FRAMES_THRESHOLD) while keeping long ones masked.

Outputs:
  data_features/<basename>_features.csv        (one row, wide columns)
  data_features/<basename>_features_long.csv   (optional: --long, tidy format)

Usage:
  python features.py --in data_angles/sample_angles.csv --outdir data_features \
    --fps 120 --angle-min 0 --angle-max 180 --vel-max 500 --spike-frames 1 \
    --active-threshold 10 --long
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============== Defaults ===============
FPS_DEFAULT = 120.0
ANGLE_MIN_DEFAULT, ANGLE_MAX_DEFAULT = 0.0, 180.0
VEL_MAX_DEFAULT = 500.0      # deg/s
SPIKE_FRAMES_THRESHOLD_DEFAULT = 1
ACTIVE_JOINTS_THRESHOLD_DEFAULT = 10.0  # deg/s


# =============== Utilities ===============
def _as_1d_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    return x[np.isfinite(x)]

def rms(x: np.ndarray) -> float:
    v = _as_1d_finite(x)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")

def std(x: np.ndarray) -> float:
    v = _as_1d_finite(x)
    return float(np.std(v, ddof=0)) if v.size else float("nan")

def coefficient_of_variation(x: np.ndarray) -> float:
    v = _as_1d_finite(x)
    if v.size == 0:
        return float("nan")
    m = np.mean(v)
    denom = abs(m)
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return float(np.std(v, ddof=0) / denom)

def zero_crossings(x: np.ndarray) -> int:
    v = _as_1d_finite(x)
    if v.size < 2:
        return 0
    s = np.sign(v)
    # count sign flips ignoring exact zeros
    return int(np.sum((np.diff(s) != 0) & (v[:-1] != 0)))

def mean_active_joints(vel_matrix: np.ndarray, threshold: float) -> float:
    # vel_matrix shape: [frames, joints]
    V = np.asarray(vel_matrix, dtype=float)
    V[~np.isfinite(V)] = np.nan
    # count active joints per frame (ignore NaNs)
    active = (np.abs(V) > threshold).astype(float)
    active[np.isnan(V)] = np.nan  # do not count masked values
    per_frame = np.nanmean(active, axis=1) * V.shape[1]  # convert fraction to count
    return float(np.nanmean(per_frame))

def mask_short_spikes(velocity: np.ndarray, max_thresh: float, max_run: int) -> np.ndarray:
    """
    Keep long |v| spikes masked (NaN). Optionally unmask short runs (<= max_run) to avoid over-masking.
    Input may already have NaNs (from earlier validation).
    """
    v = velocity.copy().astype(float)
    bad = np.abs(v) > max_thresh
    # Respect pre-existing NaNs as bad too (stay masked)
    bad |= ~np.isfinite(v)

    if max_run <= 0:
        v[bad] = np.nan
        return v

    i = 0
    n = len(bad)
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            run_len = j - i
            if run_len <= max_run:
                # unmask these short spikes ONLY if original sample was finite
                bad[i:j] = False
            i = j
        else:
            i += 1

    v[bad] = np.nan
    return v

def enforce_angle_range(angle: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    a = angle.copy().astype(float)
    bad = (~np.isfinite(a)) | (a < a_min) | (a > a_max)
    a[bad] = np.nan
    return a


# =============== Core ===============
def compute_movement_features(angle_csv: Path,
                               outdir: Path,
                               fps: float,
                               angle_min: float,
                               angle_max: float,
                               vel_max: float,
                               spike_frames: int,
                               active_threshold: float,
                               write_long: bool) -> Path:
    angle_csv = Path(angle_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(angle_csv)
    if "Time" not in df.columns:
        raise ValueError("Angle CSV must contain a 'Time' column.")

    # Detect columns produced by joint_angles.py
    angle_cols = [c for c in df.columns if c.endswith("_Angle")]
    vel_cols = [c for c in df.columns if c.endswith("_Angle_Velocity") or c.endswith("_Velocity")]
    # Pair angles to velocities by base name
    pairs: List[Dict[str, str]] = []
    for a in angle_cols:
        base = a[:-6]  # drop "_Angle"
        v_name = f"{base}_Angle_Velocity"
        if v_name not in df.columns:
            # fallback: older naming
            v_name = f"{base}_Velocity"
        if v_name in df.columns:
            pairs.append({"angle": a, "vel": v_name, "base": base})
        else:
            # skip if no velocity present
            print(f"⚠️  Skipping {a}: matching velocity column not found.")

    if not pairs:
        raise ValueError("No angle/velocity pairs found (expected '*_Angle' and '*_Angle_Velocity').")

    features_wide: Dict[str, float] = {}
    long_rows = []
    all_vel_for_active = []

    for p in pairs:
        a_name, v_name, base = p["angle"], p["vel"], p["base"]
        a = df[a_name].to_numpy(dtype=float)
        v = df[v_name].to_numpy(dtype=float)

        # Re-apply conservative masking for safety
        a_masked = enforce_angle_range(a, angle_min, angle_max)
        v_masked = mask_short_spikes(v, max_thresh=vel_max, max_run=spike_frames)

        # ---- Per-joint features ----
        # Velocity-based stats after masking
        val_rms = rms(v_masked)
        val_std = std(v_masked)
        # CV on angles after masking
        val_cv = coefficient_of_variation(a_masked)
        # ZeroCross on velocity after masking
        val_zx = zero_crossings(v_masked)

        features_wide[f"{base}_Angle_RMS"] = val_rms
        features_wide[f"{base}_Angle_STD"] = val_std
        features_wide[f"{base}_Angle_CV"] = val_cv
        features_wide[f"{base}_Angle_ZeroCross"] = val_zx

        if write_long:
            long_rows.append({
                "Signal": base,
                "Feature": "RMS",
                "Value": val_rms
            })
            long_rows.append({
                "Signal": base,
                "Feature": "STD",
                "Value": val_std
            })
            long_rows.append({
                "Signal": base,
                "Feature": "CV",
                "Value": val_cv
            })
            long_rows.append({
                "Signal": base,
                "Feature": "ZeroCross",
                "Value": val_zx
            })

        # For MeanActiveJoints
        all_vel_for_active.append(np.abs(v_masked))

    # MeanActiveJoints (uses masked velocities)
    if all_vel_for_active:
        V = np.vstack(all_vel_for_active).T  # [frames x joints]
        maj = mean_active_joints(V, threshold=active_threshold)
        features_wide["MeanActiveJoints"] = maj
    else:
        features_wide["MeanActiveJoints"] = float("nan")

    # ---- Save outputs ----
    base_name = Path(angle_csv).stem
    wide_path = outdir / f"{base_name}_features.csv"
    pd.DataFrame([features_wide]).to_csv(wide_path, index=False)

    if write_long:
        long_df = pd.DataFrame(long_rows, columns=["Signal", "Feature", "Value"])
        long_path = outdir / f"{base_name}_features_long.csv"
        long_df.to_csv(long_path, index=False)
        print(f"📝 Saved long features → {long_path}")

    print(f"✅ Feature file saved → {wide_path}")
    return wide_path


# =============== CLI ===============
def build_argparser():
    ap = argparse.ArgumentParser(description="Compute joint-level features from angle/velocity CSV.")
    ap.add_argument("--in", dest="angle_csv", type=str, required=True, help="Path to *_angles.csv from joint_angles.py")
    ap.add_argument("--outdir", type=str, default="data_features", help="Output directory")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT, help="Sampling rate (used for documentation; features are framewise).")
    ap.add_argument("--angle-min", type=float, default=ANGLE_MIN_DEFAULT, help="Minimum valid angle (deg).")
    ap.add_argument("--angle-max", type=float, default=ANGLE_MAX_DEFAULT, help="Maximum valid angle (deg).")
    ap.add_argument("--vel-max", type=float, default=VEL_MAX_DEFAULT, help="Max |angular velocity| (deg/s) for masking.")
    ap.add_argument("--spike-frames", type=int, default=SPIKE_FRAMES_THRESHOLD_DEFAULT,
                    help="Unmask short velocity spikes with run length ≤ this. Use 0 to keep all spikes masked.")
    ap.add_argument("--active-threshold", type=float, default=ACTIVE_JOINTS_THRESHOLD_DEFAULT,
                    help="Threshold (deg/s) for MeanActiveJoints.")
    ap.add_argument("--long", action="store_true", help="Also write a tidy long-format table.")
    return ap

def main():
    args = build_argparser().parse_args()
    compute_movement_features(
        angle_csv=Path(args.angle_csv),
        outdir=Path(args.outdir),
        fps=args.fps,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        vel_max=args.vel_max,
        spike_frames=args.spike_frames,
        active_threshold=args.active_threshold,
        write_long=args.long,
    )

if __name__ == "__main__":
    main()
