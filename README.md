# Motion Capture Analysis Pipeline

This repository provides a complete pipeline for processing, validating, and analyzing 3D human motion capture data. Each stage outputs CSV files containing (x, y, z) coordinates or derived features. For every CSV extraction step, you can use the included `skeleton_3d.py` and `skeleton_3d_animation.py` scripts to visualize and validate the results.

---

## Pipeline Overview

### 1. Preprocessing Script (`check_sampling.py`)

**Purpose:**  
Clean and normalize raw motion capture data for downstream analysis.

**Key Steps:**
- Interpolate missing timestamps and values
- Detrend low-frequency drift
- Apply low-pass Butterworth filter
- Resample to uniform 120 Hz
- Normalize positions relative to torso/pelvis
- Export cleaned CSV for downstream analysis

**Validation:**  
Use `skeleton_3d.py` and `skeleton_3d_animation.py` to visualize and check the cleaned data.

---

### 2. Joint Angle Computation Script (`joint_angles.py`)

**Purpose:**  
Compute biomechanically interpretable 3D joint angles from preprocessed data.

**Key Steps:**
- For each joint triplet, calculate 3D angles between segments
- Apply Butterworth low-pass and Savitzky-Golay filters
- Compute angular velocity (gradient)
- Validation checks:
    - Angle range (0°–180°)
    - Angular velocity (< 500°/s)
    - NaN/missing values
    - Spike duration reporting

**Output:**  
CSV with columns: Time, Angle, Angle_Velocity per joint.


### 3. Feature Extraction Script (`features.py`)

**Purpose:**  
Derive per-session motor diversity features from angle/velocity data.

**Features Extracted (per joint):**
- RMS (amplitude of motion)
- Standard Deviation (variability)
- Coefficient of Variation (CV) on angles
- Zero-Crossings (rhythmicity from angular velocity)
- Mean Active Joints (frame-wise count with meaningful velocity)

**Validation:**
- Ensure no feature distortion from brief spikes
- Filtering skips velocity spikes 

**Output:**  
CSV with one row per session.




