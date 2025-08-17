import pandas as pd
import numpy as np
import os
from pathlib import Path
from itertools import groupby

# === CONFIGURATION ===
FPS = 120
ANGLE_MIN, ANGLE_MAX = 0, 180
MAX_ANGLE_VELOCITY = 500
SPIKE_FRAMES_THRESHOLD = 5

# === FEATURE COMPUTATION ===
def mean_active_joints(velocities, threshold=10):
    """
    Counts how many joints are active (velocity > threshold) per frame,
    then averages across all frames.
    """
    active_counts = (np.abs(velocities) > threshold).sum(axis=1)
    return np.mean(active_counts)

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def std(x):
    return np.std(x)

def coefficient_of_variation(x):
    mean = np.mean(x)
    return np.std(x) / mean if mean != 0 else np.nan

def zero_crossings(x):
    return ((np.diff(np.sign(x)) != 0) & (x[:-1] != 0)).sum()

def mask_spikes(velocity, max_thresh=MAX_ANGLE_VELOCITY, max_run=SPIKE_FRAMES_THRESHOLD):
    mask = np.abs(velocity) > max_thresh
    clean_mask = mask.copy()

    # Identify contiguous spike regions
    i = 0
    while i < len(mask):
        if mask[i]:
            run_start = i
            while i < len(mask) and mask[i]:
                i += 1
            run_len = i - run_start
            if run_len <= max_run:
                clean_mask[run_start:i] = False  # Unmask short spikes
        else:
            i += 1
    return np.ma.masked_array(velocity, mask=clean_mask)

# === MAIN ===
def compute_movement_features(angle_csv):
    df = pd.read_csv(angle_csv)
    base_name = Path(angle_csv).stem

    time = df['Time'].values
    joint_cols = [col for col in df.columns if col.endswith('_Angle')]
    velocity_cols = [col for col in df.columns if col.endswith('_Angle_Velocity')]

    features = {}
    all_velocities = []

    for angle_col in joint_cols:
        velocity_col = angle_col + '_Velocity'

        angle = df[angle_col].values
        velocity = df[velocity_col].values
        masked_velocity = mask_spikes(velocity)

        # Individual joint features
        features[f'{angle_col}_RMS'] = rms(masked_velocity)
        features[f'{angle_col}_STD'] = std(masked_velocity)
        features[f'{angle_col}_CV'] = coefficient_of_variation(angle)
        features[f'{angle_col}_ZeroCross'] = zero_crossings(masked_velocity)

        all_velocities.append(np.abs(masked_velocity))

    # Mean Active Joints
    all_vel_matrix = np.vstack(all_velocities).T  # shape: [frames x joints]
    features['MeanActiveJoints'] = mean_active_joints(all_vel_matrix)

    # Save to CSV
    out_dir = "data_features"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_features.csv")
    pd.DataFrame([features]).to_csv(out_path, index=False)
    print(f"\n✅ Feature file saved to: {out_path}")

# === USAGE ===
if __name__ == "__main__":
    input_file = input("Enter path to joint angle CSV: ").strip()
    compute_movement_features(input_file)
