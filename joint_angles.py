#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_angles.py — compute joint angles & angular velocities, headless (no prompts)

Usage:
  python joint_angles.py --in data_preprocessed/foo_preprocessed.csv --outdir data_angles
  python joint_angles.py --in-dir data_preprocessed --outdir data_angles
"""

import argparse
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

# =========================== Config ===========================
FPS = 120.0
BUTTER_CUTOFF = 8.0      # Hz for low-pass on raw angle
BUTTER_ORDER  = 4
SAVGOL_WINDOW = 11       # odd
SAVGOL_POLY   = 2

ANGLE_MIN, ANGLE_MAX = 0.0, 180.0
VEL_THRESH = 500.0       # deg/s (validation only)

# Triplets: (proximal, joint(vertex), distal)
ANGLE_TRIPLETS = [
    ("Spine3", "LeftShoulder",  "LeftArm"),
    ("Spine3", "RightShoulder", "RightArm"),
    ("LeftShoulder",  "LeftArm",  "LeftForeArm"),
    ("RightShoulder", "RightArm", "RightForeArm"),
    ("Hips", "LeftUpLeg",  "LeftLeg"),
    ("Hips", "RightUpLeg", "RightLeg"),
    ("LeftUpLeg",  "LeftLeg",  "LeftFoot"),
    ("RightUpLeg", "RightLeg", "RightFoot"),
    ("Hips", "Spine1", "Spine3"),
    ("Spine1", "Spine3", "Head"),
]

AXES = ["X", "Y", "Z"]

# ========================= Utilities ==========================
def butter_lowpass_filter(y, cutoff=BUTTER_CUTOFF, fs=FPS, order=BUTTER_ORDER):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, y)

def apply_savgol(y, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    if window % 2 == 0:
        window += 1
    if window < 3:
        window = 3
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")

def gradient_deg(angle_deg, fps=FPS):
    # seconds spacing = 1/fps
    return np.gradient(angle_deg, 1.0/float(fps))

def get_xyz(df, joint):
    cols = [f"{joint}_Position_{a}" for a in AXES]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype=float)

def angle_between(v1, v2):
    # Safe 3D angle in degrees for each row
    v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12)
    d = np.sum(v1n * v2n, axis=1)
    d = np.clip(d, -1.0, 1.0)
    return np.degrees(np.arccos(d))

def longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    run = 0
    for m in mask:
        if m:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best

def validate_series(name, angle, velocity,
                    a_min=ANGLE_MIN, a_max=ANGLE_MAX, v_abs=VEL_THRESH):
    bad_range = (angle < a_min) | (angle > a_max)
    bad_vel   = np.abs(velocity) > v_abs

    if bad_range.any():
        count = int(bad_range.sum())
        run   = longest_true_run(bad_range)
        print(f"⚠️  {name}: angle out of [{a_min},{a_max}] for {count} frames (longest run {run})")

    if bad_vel.any():
        count = int(bad_vel.sum())
        run   = longest_true_run(bad_vel)
        print(f"⚠️  {name}: |angular velocity|>{v_abs}°/s for {count} frames (longest run {run})")
        if run > 25:
            print(f"🚨 {name}: long velocity run > {v_abs}°/s (run={run}) — inspect input or filtering.")

    if (not bad_range.any()) and (not bad_vel.any()) and np.isfinite(angle).all() and np.isfinite(velocity).all():
        print(f"✅ Validated: {name}")

# ========================= Core logic =========================
def process_file(in_csv: Path, outdir: Path, fps: float = FPS):
    df = pd.read_csv(in_csv)
    # Time column (not required; we can synthesize)
    if "Time" in df.columns:
        time = df["Time"].to_numpy(dtype=float)
    else:
        time = np.arange(len(df), dtype=float) / float(fps)

    out = {"Time": time}

    for (prox, joint, dist) in ANGLE_TRIPLETS:
        P = get_xyz(df, prox)
        J = get_xyz(df, joint)
        D = get_xyz(df, dist)
        base = f"{prox}_{joint}_{dist}_Angle"

        if P is None or J is None or D is None:
            print(f"⚠️  Missing columns for triplet {prox}-{joint}-{dist}; skipping.")
            continue

        # Build vectors at the joint (vertex)
        v1 = J - P
        v2 = D - J

        # Raw angle in degrees
        ang_raw = angle_between(v1, v2)

        # Gentle denoise for numerical derivative stability (low-pass + Savitzky–Golay)
        try:
            ang_f = butter_lowpass_filter(ang_raw, cutoff=BUTTER_CUTOFF, fs=fps, order=BUTTER_ORDER)
            ang_s = apply_savgol(ang_f, window=SAVGOL_WINDOW, poly=SAVGOL_POLY)
        except Exception:
            # fallback: at least ensure finite
            ang_s = np.asarray(ang_raw, dtype=float)

        vel = gradient_deg(ang_s, fps=fps)

        out[base] = ang_s
        out[f"{base}_Velocity"] = vel

        # Validation only (no masking here; masking is in features.py)
        validate_series(base, ang_s, vel, a_min=ANGLE_MIN, a_max=ANGLE_MAX, v_abs=VEL_THRESH)

    # Save
    outdir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out)
    out_csv = outdir / f"{in_csv.stem}_angles.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Angles saved: {out_csv}")
    return out_csv

# =========================== CLI ===========================
def build_argparser():
    ap = argparse.ArgumentParser(description="Compute joint angles & angular velocity from preprocessed XYZ.")
    ap.add_argument("--in", dest="in_path", type=str, help="Single input CSV (preprocessed).")
    ap.add_argument("--in-dir", dest="in_dir", type=str, help="Directory of preprocessed CSVs.")
    ap.add_argument("--outdir", type=str, default="data_angles", help="Output directory.")
    ap.add_argument("--fps", type=float, default=FPS, help="Sampling rate (Hz) for velocity calculation.")
    ap.add_argument("--cutoff", type=float, default=BUTTER_CUTOFF, help="Butterworth cutoff (Hz).")
    ap.add_argument("--order", type=int, default=BUTTER_ORDER, help="Butterworth order.")
    ap.add_argument("--savgol-window", type=int, default=SAVGOL_WINDOW, help="Savitzky–Golay window (odd).")
    ap.add_argument("--savgol-poly", type=int, default=SAVGOL_POLY, help="Savitzky–Golay polyorder.")
    ap.add_argument("--vel-thresh", type=float, default=VEL_THRESH, help="Validation |angular velocity| threshold (deg/s).")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    # allow runtime overrides
    global FPS, BUTTER_CUTOFF, BUTTER_ORDER, SAVGOL_WINDOW, SAVGOL_POLY, VEL_THRESH
    FPS = float(args.fps)
    BUTTER_CUTOFF = float(args.cutoff)
    BUTTER_ORDER = int(args.order)
    SAVGOL_WINDOW = int(args.savgol_window)
    SAVGOL_POLY = int(args.savgol_poly)
    VEL_THRESH = float(args.vel_thresh)

    outdir = Path(args.outdir)
    files = []
    if args.in_path:
        files.append(Path(args.in_path))
    if args.in_dir:
        files += sorted(Path(args.in_dir).glob("*.csv"))
    if not files:
        sys.exit("Provide --in <file.csv> or --in-dir <dir>")

    ok = 0
    for f in files:
        try:
            process_file(f, outdir, fps=FPS)
            ok += 1
        except Exception as e:
            print(f"❌ Failed {f.name}: {e}")

    if ok == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
