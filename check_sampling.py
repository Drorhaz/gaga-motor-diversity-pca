import pandas as pd
import numpy as np
import os
import sys
from scipy.interpolate import interp1d, PchipInterpolator
from datetime import datetime
from pathlib import Path

# ========== CONFIGURATION ==========
REQUIRED_MARKERS = [
    'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'Spine3', 'Spine1', 'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'RightUpLeg', 'RightLeg', 'RightFoot'
]
AXES = ['X', 'Y', 'Z']
REQUIRED_COLUMNS = [f"{marker}_Position_{axis}" for marker in REQUIRED_MARKERS for axis in AXES]
TIME_SCALE_FACTOR = 0.00104166  # Converts timestamp ticks to seconds (~1.04166 ms)
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


def main():
    filepath = input("Enter path to CSV file: ").strip()
    if not os.path.exists(filepath):
        print("❌ File does not exist.")
        sys.exit(1)

    print(f"\n📂 Loaded: {filepath}\n")
    df = pd.read_csv(filepath)

    # ---- Check columns ----
    print("Checking required columns...")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if "TimeStamp" not in df.columns:
        print("❌ Missing required column: TimeStamp")
        sys.exit(1)
    if missing_cols:
        print(f"WARNING: Missing marker columns: {missing_cols}\n")
    else:
        print("✅ All required marker columns present.\n")

    # ---- Detect flat segments ----
    print("Detecting flat segments...")
    flat_segments_map = {}
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            segments = detect_flat_segments(df[col].values)
            if segments:
                flat_segments_map[col] = segments

    if flat_segments_map:
        print("⚠️  Flat segments found:")
        for col, segs in flat_segments_map.items():
            print(f"  - {col}: {len(segs)} segments")
    else:
        print("✅ No flat segments detected.")

    # ---- Sampling Check ----
    ts_raw = df['TimeStamp'].values
    ts_sec = ts_raw * TIME_SCALE_FACTOR
    delta_ts = np.diff(ts_sec)
    median_dt = np.median(delta_ts)
    evaluated_fps = 1 / median_dt if median_dt > 0 else 0
    actual_duration = ts_sec[-1] - ts_sec[0]
    expected_duration = len(df) / DEFAULT_EXPECTED_FPS
    evaluated_duration = len(df) / evaluated_fps if evaluated_fps > 0 else 0
    drift_expected = abs((actual_duration - expected_duration) / expected_duration) * 100
    drift_evaluated = abs((actual_duration - evaluated_duration) / evaluated_duration) * 100

    print("\n🔍 Sampling Check:")
    print(f"  - Median Δt:       {median_dt*1000:.4f} ms")
    print(f"  - Evaluated FPS:   {evaluated_fps:.2f} Hz")
    print(f"  - Expected FPS:    {DEFAULT_EXPECTED_FPS}")
    print(f"  - Expected Dur:    {expected_duration:.2f} s")
    print(f"  - Actual Dur:      {actual_duration:.2f} s")
    print(f"  - Drift (Expected): {drift_expected:.4f} %")
    print(f"  - Drift (Evaluated): {drift_evaluated:.4f} %\n")

    if drift_evaluated > 2.0:
        print("❗ Clock drift exceeds 2% — resampling recommended.\n")

    # ---- Ask resample rate ----
    try:
        target_fps = float(input(f"Enter desired FPS to resample to (default {DEFAULT_EXPECTED_FPS}): ") or DEFAULT_EXPECTED_FPS)
    except ValueError:
        target_fps = DEFAULT_EXPECTED_FPS

    # ---- Interpolation method ----
    method = input("Interpolation method [cubic/linear/pchip] (default cubic): ").strip().lower() or 'cubic'
    if method not in ['linear', 'cubic', 'pchip']:
        print("⚠️ Invalid method. Defaulting to cubic.")
        method = 'cubic'

    if method == 'cubic' and flat_segments_map:
        print("⚠️ Cubic interpolation may overshoot with flat segments. Consider using linear or pchip.")

    # ---- Resample ----
    num_frames = int(np.floor((ts_sec[-1] - ts_sec[0]) * target_fps))
    new_time = np.linspace(ts_sec[0], ts_sec[-1], num=num_frames)
    interpolated = {}

    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            y = df[col].values
            if np.isnan(y).all():
                continue
            mask = ~np.isnan(y)
            try:
                if method == 'pchip':
                    interp_func = PchipInterpolator(ts_sec[mask], y[mask])
                else:
                    interp_func = interp1d(ts_sec[mask], y[mask], kind=method, fill_value="extrapolate")
                interpolated[col] = interp_func(new_time)
            except Exception as e:
                print(f"❌ Could not interpolate {col}: {e}")

    interpolated['Time'] = new_time
    df_interp = pd.DataFrame(interpolated)

    # ---- Restore Flat Segments ----
    print("\n🔁 Restoring flat segments...")
    for col, segments in flat_segments_map.items():
        if col in df_interp.columns:
            for start, end, value in segments:
                t_start, t_end = ts_sec[start], ts_sec[end]
                mask = (df_interp['Time'] >= t_start) & (df_interp['Time'] <= t_end)
                df_interp.loc[mask, col] = value

    # ---- Save result ----
    save_dir = input("Enter output folder path (default: ./data_clean): ").strip() or "data_clean"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/{Path(filepath).stem}_resampled_{int(target_fps)}Hz.csv"
    df_interp.to_csv(out_path, index=False)
    print(f"\n✅ Saved cleaned data to: {out_path}\n")


if __name__ == '__main__':
    main()


