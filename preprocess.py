import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from pathlib import Path

# === CONFIGURATION ===
CUTOFF_HZ = 8
ORDER = 4
EXPECTED_SAMPLING_RATE = 120  # Hz (for filtering)
DATA_DIR = "data_clean"
OUTPUT_DIR = "data_preprocessed"
REQUIRED_AXES = ['X', 'Y', 'Z']
SHOULDER_LEFT = 'LeftShoulder'
SHOULDER_RIGHT = 'RightShoulder'
HIP_LEFT = 'LeftUpLeg'
HIP_RIGHT = 'RightUpLeg'
SPINE = 'Spine1'
HIP_CENTER = 'Hips'


def butterworth_filter(data, fps, cutoff=CUTOFF_HZ, order=ORDER):
    b, a = butter(order, cutoff / (0.5 * fps), btype='low')
    return filtfilt(b, a, data, axis=0)


def extract_joint_columns(df, joint_name):
    return [f"{joint_name}_Position_{axis}" for axis in REQUIRED_AXES]


def normalize_view(df):
    print("\n🔄 Step 1: Viewpoint normalization (Ajili et al.)")
    df_normalized = df.copy()
    joint_names = set(col.split('_Position_')[0] for col in df.columns if '_Position_' in col)

    for idx, row in df.iterrows():
        try:
            hip_pos = row[extract_joint_columns(df, HIP_CENTER)].values

            l_hip = row[extract_joint_columns(df, HIP_LEFT)].values
            r_hip = row[extract_joint_columns(df, HIP_RIGHT)].values
            x_axis = l_hip - r_hip
            x_axis /= np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else 1

            mid_hip = (l_hip + r_hip) / 2
            spine = row[extract_joint_columns(df, SPINE)].values
            y_axis = spine - mid_hip
            y_axis /= np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0 else 1

            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else 1

            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0 else 1

            R = np.vstack([x_axis, y_axis, z_axis])

            for joint in joint_names:
                cols = extract_joint_columns(df, joint)
                if not all(col in df.columns for col in cols):
                    continue
                joint_vec = row[cols].values - hip_pos
                rotated = R @ joint_vec
                df_normalized.loc[idx, cols] = rotated
        except Exception as e:
            print(f"⚠️ Frame {idx}: View normalization failed — {e}")
            continue

    print("✅ Viewpoint normalization complete.")
    return df_normalized


def size_normalize(df, shoulder_width_m):
    print(f"\n🔄 Step 2: Size normalization using shoulder width = {shoulder_width_m:.3f} meters")
    df_scaled = df.copy()
    joint_names = set(col.split('_Position_')[0] for col in df.columns if '_Position_' in col)

    for joint in joint_names:
        cols = extract_joint_columns(df, joint)
        if not all(col in df.columns for col in cols):
            continue
        df_scaled[cols] = df_scaled[cols] / shoulder_width_m

    print("✅ Size normalization complete.")
    return df_scaled


def apply_filter(df, fps):
    print("\n🔃 Step 3: Low-pass filtering (Butterworth 4th order, 8Hz)")
    df_filtered = df.copy()
    for col in df.columns:
        if '_Position_' in col and df[col].dtype in [np.float64, np.float32, np.float16]:
            df_filtered[col] = butterworth_filter(df[[col]].values.flatten(), fps)
    print("✅ Filtering complete.")
    return df_filtered


def preprocess_motion_data(file_path, shoulder_width_m, fps=EXPECTED_SAMPLING_RATE):
    print(f"\n🚀 Preprocessing started for: {file_path}")
    df = pd.read_csv(file_path)
    base_name = Path(file_path).stem

    # Step 1: Viewpoint normalization
    df_local = normalize_view(df)

    # Step 2: Size normalization
    df_normalized = size_normalize(df_local, shoulder_width_m)

    # Step 3: Filtering
    df_filtered = apply_filter(df_normalized, fps)

    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_preprocessed.csv")
    df_filtered.to_csv(out_path, index=False)
    print(f"\n✅ Preprocessed data saved to: {out_path}")


# === ENTRY POINT ===
if __name__ == "__main__":
    print("🔧 Preprocessing motion data")

    filename = input("Enter filename from data_clean/: ").strip()
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print("❌ File not found.")
        exit(1)

    try:
        shoulder_width_m = float(input("Enter participant shoulder width in meters (e.g., 0.39): ").strip())
    except ValueError:
        print("❌ Invalid shoulder width.")
        exit(1)

    preprocess_motion_data(file_path, shoulder_width_m)
