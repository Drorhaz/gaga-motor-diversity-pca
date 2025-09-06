#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d, PchipInterpolator
from pathlib import Path

# ====== CONFIG (column schema) ======
REQUIRED_MARKERS = [
    'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'Spine3', 'Spine1', 'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'RightUpLeg', 'RightLeg', 'RightFoot'
]
AXES = ['X', 'Y', 'Z']
REQUIRED_COLUMNS = [f"{m}_Position_{a}" for m in REQUIRED_MARKERS for a in AXES]

DEFAULT_EXPECTED_FPS = 120
FLAT_SEGMENT_MIN_LENGTH = 3


def detect_flat_segments(series, eps=1e-6):
    flat_segments = []
    run_value = None
    run_start = None
    run_length = 0

    for i in range(1, len(series)):
        if pd.isna(series[i]) or pd.isna(series[i - 1]):
            if run_length >= FLAT_SEGMENT_MIN_LENGTH:
                flat_segments.append((run_start, i - 1, run_value))
            run_start, run_length = None, 0
            continue

        if abs(series[i] - series[i - 1]) < eps:
            if run_start is None:
                run_start = i - 1
                run_value = series[i]
            run_length += 1
        else:
            if run_length >= FLAT_SEGMENT_MIN_LENGTH:
                flat_segments.append((run_start, i - 1, run_value))
            run_start, run_length = None, 0

    if run_length >= FLAT_SEGMENT_MIN_LENGTH:
        flat_segments.append((run_start, len(series) - 1, run_value))

    return flat_segments


def resample_file(
    filepath: Path,
    out_dir: Path,
    target_fps: float = DEFAULT_EXPECTED_FPS,
    method: str = "pchip",
    tick_scale_sec: float = 0.00104166,
    plots: bool = False,
):
    """
    Check timing, (optionally) resample to uniform grid, and write cleaned file.
    Returns Path to the written CSV.
    """
    filepath = Path(filepath)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loaded: {filepath}\n", flush=True)
    df = pd.read_csv(filepath)

    # ---- Column checks ----
    print("Checking required columns...", flush=True)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if "TimeStamp" not in df.columns:
        raise ValueError("Missing required column: TimeStamp")
    if missing_cols:
        print(f"⚠️  Missing marker columns: {missing_cols}\n", flush=True)
    else:
        print("✅ All required marker columns present.\n", flush=True)

    # ---- Detect flat segments ----
    print("Detecting flat segments...", flush=True)
    flat_segments_map = {}
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            segs = detect_flat_segments(df[col].values)
            if segs:
                flat_segments_map[col] = segs

    if flat_segments_map:
        print("⚠️  Flat segments found:", flush=True)
        for col, segs in flat_segments_map.items():
            print(f"  - {col}: {len(segs)} segments", flush=True)
    else:
        print("✅ No flat segments detected.", flush=True)

    # ---- Timing diagnostics ----
    ts_raw = df["TimeStamp"].to_numpy()
    ts_sec = ts_raw * float(tick_scale_sec)
    delta_ts = np.diff(ts_sec)
    median_dt = float(np.median(delta_ts)) if len(delta_ts) else np.nan
    evaluated_fps = (1.0 / median_dt) if (median_dt and median_dt > 0) else np.nan
    actual_duration = float(ts_sec[-1] - ts_sec[0]) if len(ts_sec) else np.nan
    expected_duration = len(df) / DEFAULT_EXPECTED_FPS
    evaluated_duration = len(df) / evaluated_fps if (evaluated_fps and evaluated_fps > 0) else np.nan
    drift_expected = abs((actual_duration - expected_duration) / expected_duration) * 100 if expected_duration else np.nan
    drift_evaluated = abs((actual_duration - evaluated_duration) / evaluated_duration) * 100 if evaluated_duration else np.nan

    print("\n🔍 Sampling Check:", flush=True)
    print(f"  - Median Δt:        {median_dt*1000:.4f} ms" if np.isfinite(median_dt) else "  - Median Δt:  NA", flush=True)
    print(f"  - Evaluated FPS:    {evaluated_fps:.2f} Hz" if np.isfinite(evaluated_fps) else "  - Evaluated FPS: NA", flush=True)
    print(f"  - Expected FPS:     {DEFAULT_EXPECTED_FPS}", flush=True)
    print(f"  - Expected Dur:     {expected_duration:.2f} s", flush=True)
    print(f"  - Actual Dur:       {actual_duration:.2f} s" if np.isfinite(actual_duration) else "  - Actual Dur: NA", flush=True)
    print(f"  - Drift (Expected): {drift_expected:.4f} %" if np.isfinite(drift_expected) else "  - Drift (Expected): NA", flush=True)
    print(f"  - Drift (Evaluated): {drift_evaluated:.4f} %" if np.isfinite(drift_evaluated) else "  - Drift (Evaluated): NA", flush=True)

    if np.isfinite(drift_evaluated) and drift_evaluated > 2.0:
        print("❗ Clock drift exceeds 2% — resampling recommended.\n", flush=True)

    # ---- Interpolation method guard ----
    method = (method or "pchip").lower()
    if method not in ("linear", "cubic", "pchip"):
        print("⚠️ Invalid method. Defaulting to pchip.", flush=True)
        method = "pchip"
    if method == "cubic" and flat_segments_map:
        print("⚠️  Cubic interpolation may overshoot with flat segments. Consider linear or pchip.", flush=True)

    # ---- Uniform grid & interpolation ----
    num_frames = int(np.floor((ts_sec[-1] - ts_sec[0]) * target_fps))
    if num_frames < 2:
        # Write passthrough if too short
        out_path = out_dir / f"{filepath.stem}_uniform_{int(target_fps)}Hz.csv"
        df_out = df.copy()
        df_out["Time"] = ts_sec
        df_out.to_csv(out_path, index=False)
        print(f"\n⚠️ Too few samples; wrote passthrough: {out_path}\n", flush=True)
        return out_path

    new_time = np.linspace(ts_sec[0], ts_sec[-1], num=num_frames)
    interpolated = {}

    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            y = df[col].to_numpy(dtype=float)
            if np.isnan(y).all():
                continue
            mask = ~np.isnan(y)
            try:
                if method == "pchip":
                    interp_func = PchipInterpolator(ts_sec[mask], y[mask])
                else:
                    interp_func = interp1d(ts_sec[mask], y[mask], kind=method, fill_value="extrapolate", assume_sorted=True)
                interpolated[col] = interp_func(new_time)
            except Exception as e:
                print(f"❌ Could not interpolate {col}: {e}", flush=True)

    interpolated["Time"] = new_time
    df_interp = pd.DataFrame(interpolated)

    # ---- Restore flat segments exactly (optional but recommended) ----
    if flat_segments_map:
        print("\n🔁 Restoring flat segments...", flush=True)
        for col, segments in flat_segments_map.items():
            if col in df_interp.columns:
                for start, end, value in segments:
                    t_start, t_end = ts_sec[start], ts_sec[end]
                    m = (df_interp["Time"] >= t_start) & (df_interp["Time"] <= t_end)
                    df_interp.loc[m, col] = value

    # ---- Save result ----
    out_path = out_dir / f"{filepath.stem}_resampled_{int(target_fps)}Hz.csv"
    df_interp.to_csv(out_path, index=False)
    print(f"\n✅ Saved cleaned data to: {out_path}\n", flush=True)
    return out_path


def build_argparser():
    ap = argparse.ArgumentParser(description="Check/diagnose sampling and resample to a uniform grid.")
    ap.add_argument("--in", dest="in_csv", type=Path, required=True, help="Input raw CSV path.")
    ap.add_argument("--outdir", type=Path, default=Path("data_clean"), help="Output directory.")
    ap.add_argument("--target-fps", type=float, default=DEFAULT_EXPECTED_FPS, help="Target output FPS.")
    ap.add_argument("--method", type=str, default="pchip", choices=["pchip", "linear", "cubic"], help="Interpolation method.")
    # extra flags accepted by runner.py (logged but not strictly used here)
    ap.add_argument("--timebase", type=str, default="auto", help="auto|frame|timestamp (informational).")
    ap.add_argument("--tolerance", type=float, default=0.001, help="FPS tolerance for auto decision (informational).")
    ap.add_argument("--timescale", type=str, default="auto", help="auto|seconds|ms|ticks (informational).")
    ap.add_argument("--tick-scale", type=float, default=0.00104166, help="Seconds per tick when converting TimeStamp.")
    ap.add_argument("--plots", action="store_true", help="(Optional) Save diagnostic plots (not implemented).")
    return ap


def main():
    args = build_argparser().parse_args()

    if not args.in_csv.exists():
        raise SystemError(f"File does not exist: {args.in_csv}")

    print(f"ℹ️  timebase={args.timebase} tolerance={args.tolerance} timescale={args.timescale}", flush=True)
    resample_file(
        filepath=args.in_csv,
        out_dir=args.outdir,
        target_fps=args.target_fps,
        method=args.method,
        tick_scale_sec=args.tick_scale,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
