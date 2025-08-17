#!/usr/bin/env python3
"""
3D Skeleton Animation Generator (GIF)

- Prompts for CSV path and frame range (indices, inclusive)
- Auto-detects joints: <Joint>_Position_X|Y|Z
- Uses a default humanoid skeleton edge list
- Infers FPS from Time column if possible, else uses provided/default FPS
- Exports a GIF animation

Usage:
    python skeleton_animator.py
"""

import os
import re
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

def detect_joints(columns):
    coord_cols = [c for c in columns if re.search(r"_Position_[XYZ]$", c)]
    joints = sorted({c.split("_Position_")[0] for c in coord_cols})
    if not joints:
        raise ValueError("No joints found. Expected columns like <Joint>_Position_X|Y|Z.")
    return joints

def default_edges(joints):
    # Only keep edges whose joints are present
    candidates = [
        ("Hips", "Spine1"), ("Spine1", "Spine3"), ("Spine3", "Head"),
        ("Spine3", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
        ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
        ("Spine3", "RightShoulder"), ("RightShoulder", "RightArm"),
        ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
        ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
        ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
    ]
    return [(a, b) for (a, b) in candidates if a in joints and b in joints]

def infer_fps_from_time(df):
    if "Time" not in df.columns:
        return None
    t = df["Time"].astype(float).values
    if len(t) < 3:
        return None
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return None
    dt = np.median(diffs)
    if dt <= 0:
        return None
    fps = 1.0 / dt
    # Cap to a reasonable range in case of weird metadata
    if fps < 5 or fps > 1000:
        return None
    return float(fps)

def get_coords(row, joints):
    coords = {}
    for j in joints:
        coords[j] = np.array([
            row[f"{j}_Position_X"],
            row[f"{j}_Position_Y"],
            row[f"{j}_Position_Z"],
        ], dtype=float)
    return coords

def compute_equal_limits(df, joints, frame_indices):
    # Collect all points across the chosen frames
    pts = []
    for idx in frame_indices:
        row = df.iloc[idx]
        for j in joints:
            pts.append([
                row[f"{j}_Position_X"],
                row[f"{j}_Position_Y"],
                row[f"{j}_Position_Z"],
            ])
    pts = np.array(pts, dtype=float)
    x_min, x_max = float(np.min(pts[:,0])), float(np.max(pts[:,0]))
    y_min, y_max = float(np.min(pts[:,1])), float(np.max(pts[:,1]))
    z_min, z_max = float(np.min(pts[:,2])), float(np.max(pts[:,2]))
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    # Avoid zero-size box
    if max_range == 0:
        max_range = 1.0
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    return (x_mid - max_range/2, x_mid + max_range/2), \
           (y_mid - max_range/2, y_mid + max_range/2), \
           (z_mid - max_range/2, z_mid + max_range/2)

def main():
    try:
        csv_path = input("Enter CSV file path: ").strip().strip('"').strip("'")
        if not os.path.isfile(csv_path):
            print("File not found:", csv_path)
            sys.exit(1)

        start_str = input("Enter start frame index (0-based, inclusive): ").strip()
        end_str   = input("Enter end frame index (inclusive): ").strip()

        try:
            start_idx = int(start_str)
            end_idx   = int(end_str)
        except Exception:
            print("Start/end must be integers.")
            sys.exit(1)

        if end_idx < start_idx:
            print("End index must be >= start index.")
            sys.exit(1)

        fps_in = input("Optional FPS (press Enter to auto/infer): ").strip()
        out_path = input("Optional output path (e.g., /path/to/out.gif). Press Enter for default: ").strip()

        print("Loading CSV...")
        df = pd.read_csv(csv_path)
        nrows = len(df)
        if start_idx < 0 or end_idx >= nrows:
            print(f"Frame range out of bounds. CSV has {nrows} rows.")
            sys.exit(1)

        print("Detecting joints...")
        joints = detect_joints(df.columns.tolist())
        edges  = default_edges(joints)

        # Determine FPS
        if fps_in:
            try:
                fps = float(fps_in)
                if fps <= 0:
                    raise ValueError
            except Exception:
                print("Invalid FPS. Use a positive number (e.g., 120).")
                sys.exit(1)
        else:
            fps = infer_fps_from_time(df) or 120.0

        # Resolve output path
        if not out_path:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            out_path = os.path.join(
                os.path.dirname(csv_path),
                f"{base}_frames_{start_idx}_{end_idx}.gif"
            )

        frame_indices = list(range(start_idx, end_idx + 1))

        print("Precomputing axis limits...")
        xlim, ylim, zlim = compute_equal_limits(df, joints, frame_indices)

        print("Setting up figure...")
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Skeleton Animation (frames {start_idx}–{end_idx})")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)

        # Artists
        joint_xyz = np.zeros((len(joints), 3))
        scatter = ax.scatter(joint_xyz[:,0], joint_xyz[:,1], joint_xyz[:,2])
        edge_lines = []
        for _ in edges:
            line, = ax.plot([], [], [])  # default color/style
            edge_lines.append(line)
        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        def update(frame_idx):
            row = df.iloc[frame_idx]
            coords = get_coords(row, joints)
            pts = np.array([coords[j] for j in joints], dtype=float)
            scatter._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
            for (e, (a, b)) in enumerate(edges):
                A, B = coords[a], coords[b]
                edge_lines[e].set_data([A[0], B[0]], [A[1], B[1]])
                edge_lines[e].set_3d_properties([A[2], B[2]])
            if "Time" in df.columns:
                time_text.set_text(f"Time: {float(df.loc[frame_idx, 'Time']):.6f} s (frame {frame_idx})")
            else:
                time_text.set_text(f"Frame {frame_idx}")
            return [scatter, *edge_lines, time_text]

        print(f"Rendering GIF at {fps:.2f} FPS...")
        anim = animation.FuncAnimation(
            fig, update, frames=frame_indices, blit=False, interval=1000.0/fps
        )
        # PillowWriter ships with Matplotlib, no ImageMagick needed
        writer = animation.PillowWriter(fps=int(round(fps)))
        anim.save(out_path, writer=writer)
        plt.close(fig)

        print("Saved:", out_path)

    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
