#!/usr/bin/env python3
"""
Plot a range of frames from a motion-capture CSV as 3D skeletons on one graph.

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ---------- INPUTS ----------
csv_path = input("Enter CSV file path: ").strip()
frame_start = int(input("Enter starting frame index (0-based): ").strip())
frame_end = int(input("Enter ending frame index (inclusive): ").strip())

# ---------- LOAD DATA ----------
df = pd.read_csv(csv_path)

# Detect joints automatically
coord_cols = [c for c in df.columns if re.search(r"_Position_[XYZ]$", c)]
joints = sorted({c.split("_Position_")[0] for c in coord_cols})

# Default edges for a humanoid skeleton
edges = [
    ("Hips", "Spine1"), ("Spine1", "Spine3"), ("Spine3", "Head"),
    ("Spine3", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine3", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
]

# ---------- PLOT ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
colors = plt.cm.viridis(np.linspace(0, 1, frame_end - frame_start + 1))

all_pts = []

for i, frame_idx in enumerate(range(frame_start, frame_end + 1)):
    row = df.iloc[frame_idx]
    coords = {
        j: np.array([
            row[f"{j}_Position_X"], 
            row[f"{j}_Position_Y"], 
            row[f"{j}_Position_Z"]
        ], dtype=float)
        for j in joints
        if f"{j}_Position_X" in df.columns
    }
    pts = np.array(list(coords.values()))
    all_pts.extend(coords.values())

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], label=f"Frame {frame_idx}")
    for a, b in edges:
        if a in coords and b in coords:
            A = coords[a]
            B = coords[b]
            ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color=colors[i])

# Equal aspect ratio
all_pts = np.array(all_pts)
x_limits = [np.min(all_pts[:,0]), np.max(all_pts[:,0])]
y_limits = [np.min(all_pts[:,1]), np.max(all_pts[:,1])]
z_limits = [np.min(all_pts[:,2]), np.max(all_pts[:,2])]
max_range = max(x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0])
x_middle = np.mean(x_limits)
y_middle = np.mean(y_limits)
z_middle = np.mean(z_limits)
ax.set_xlim(x_middle - max_range/2, x_middle + max_range/2)
ax.set_ylim(y_middle - max_range/2, y_middle + max_range/2)
ax.set_zlim(z_middle - max_range/2, z_middle + max_range/2)

ax.set_title(f"3D Skeletons - Frames {frame_start} to {frame_end}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Save figure
out_path = os.path.join(os.path.dirname(csv_path), f"skeleton_frames_{frame_start}_{frame_end}.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot to: {out_path}")
