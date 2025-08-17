import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from pathlib import Path
import os

# === CONFIGURATION ===
FPS = 120
BUTTER_CUTOFF = 8  # Hz
BUTTER_ORDER = 4
SAVGOL_WINDOW = 11  # Must be odd
SAVGOL_POLY = 2
MAX_ANGLE_VELOCITY = 500  # deg/s
ANGLE_MIN, ANGLE_MAX = 0, 180

# === ANGLE TRIPLETS ===
ANGLE_TRIPLETS = [
    ("Spine3", "LeftShoulder", "LeftArm"),
    ("Spine3", "RightShoulder", "RightArm"),
    ("LeftShoulder", "LeftArm", "LeftForeArm"),
    ("RightShoulder", "RightArm", "RightForeArm"),
    ("Hips", "LeftUpLeg", "LeftLeg"),
    ("Hips", "RightUpLeg", "RightLeg"),
    ("LeftUpLeg", "LeftLeg", "LeftFoot"),
    ("RightUpLeg", "RightLeg", "RightFoot"),
    ("Hips", "Spine1", "Spine3"),
    ("Spine1", "Spine3", "Head")
]

# === FILTERS ===
def butter_lowpass_filter(data, cutoff=BUTTER_CUTOFF, fs=FPS, order=BUTTER_ORDER):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

def apply_savgol_filter(data, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    return savgol_filter(data, window_length=window, polyorder=poly)

# === ANGLE MATH ===
def extract_xyz(df, joint):
    return df[[f"{joint}_Position_X", f"{joint}_Position_Y", f"{joint}_Position_Z"]].values

def compute_3d_angle(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
    dot_prod = np.sum(v1_norm * v2_norm, axis=1)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    return np.degrees(np.arccos(dot_prod))

def compute_angular_velocity(angle_series, fps=FPS):
    velocity = np.gradient(angle_series, 1 / fps)
    return velocity

# === VALIDATION ===
from itertools import groupby

def validate_angle_series(angle_name, angle, velocity, angle_min=ANGLE_MIN, angle_max= ANGLE_MAX, velocity_thresh=MAX_ANGLE_VELOCITY):
    fails = {}
    
    # === Angle range check ===
    angle_violations = (angle < angle_min) | (angle > angle_max)
    if np.any(angle_violations):
        count = np.sum(angle_violations)
        longest_run = max((sum(1 for _ in g) for k, g in groupby(angle_violations) if k), default=0)
        fails["range"] = True
        print(f"⚠️ {angle_name}: angle out of range [{angle_min}–{angle_max}] — {count} frames, max run = {longest_run}")
    
    # === Angular velocity check ===
    velocity_violations = np.abs(velocity) > velocity_thresh
    if np.any(velocity_violations):
        count = np.sum(velocity_violations)
        longest_run = max((sum(1 for _ in g) for k, g in groupby(velocity_violations) if k), default=0)
        fails["velocity"] = True
        print(f"⚠️ {angle_name}: angular velocity > {velocity_thresh} deg/s — {count} frames, max run = {longest_run}")
        max_run = longest_run
        if max_run > 25:
            print(f"🚨 {angle_name}: long angular velocity spike > {MAX_ANGLE_VELOCITY} deg/s — {count} frames, max run = {max_run} ⚠️ Consider investigating.")
    
    # === NaN check ===
    if np.any(pd.isna(angle)) or np.any(pd.isna(velocity)):
        fails["NaN"] = True
        print(f"⚠️ {angle_name}: contains NaN values")
    
    if not fails:
        print(f"✅ Computed & validated: {angle_name}")
    
    return fails


# === MAIN ===
def compute_joint_angles(csv_path):
    print(f"\n📁 Loading preprocessed data from: {csv_path}")
    df = pd.read_csv(csv_path)
    time = np.arange(len(df)) / FPS
    out_df = pd.DataFrame({"Time": time})

    for prox, joint, dist in ANGLE_TRIPLETS:
        try:
            v1 = extract_xyz(df, joint) - extract_xyz(df, prox)
            v2 = extract_xyz(df, dist) - extract_xyz(df, joint)

            angle_raw = compute_3d_angle(v1, v2)
            angle_butter = butter_lowpass_filter(angle_raw)
            angle_savgol = apply_savgol_filter(angle_butter)
            angle_velocity = compute_angular_velocity(angle_savgol)

            base_col = f"{prox}_{joint}_{dist}_Angle"
            out_df[f"{base_col}"] = angle_savgol
            out_df[f"{base_col}_Velocity"] = angle_velocity

            # Validation
            validate_angle_series(base_col, angle_savgol, angle_velocity)

            print(f"✅ Computed & validated: {base_col}")

        except Exception as e:
            print(f"❌ Failed {prox}-{joint}-{dist}: {e}")

    # Save result
    out_dir = "data_angles"
    os.makedirs(out_dir, exist_ok=True)
    base_name = Path(csv_path).stem
    out_path = os.path.join(out_dir, f"{base_name}_angles.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Angles saved to: {out_path}")

# === USAGE ===
if __name__ == "__main__":
    input_file = input("Enter path to preprocessed CSV: ").strip()
    compute_joint_angles(input_file)
