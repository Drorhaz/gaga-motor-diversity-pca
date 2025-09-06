#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runner.py — config-driven pipeline with verbose, stage-by-stage status prints,
and the ability to resume from any stage.

Stages:
  S0) discover raw CSVs
  S1) check_sampling.py       → data_clean
  S2) preprocess.py           → data_preprocessed
  S3) joint_angles.py         → data_angles
  S4) features.py             → data_features
  S5) unify                   → unified_features.csv
  S6) PCA.py (headless)       → mdi_out
  S7) compute_deltas.py       → mdi_out
  S8) paired_mdi_stats.py     → mdi_out
  S9) lopo_logistic.py        → mdi_out
"""

import sys
import re
import time
import argparse
import subprocess
from pathlib import Path

try:
    import yaml
except Exception:
    print("❌ Missing dependency: pyyaml. Install with: pip install pyyaml", flush=True)
    sys.exit(1)

import pandas as pd


# --------------------- Pretty logging ---------------------
def banner(title: str):
    line = "═" * max(10, len(title) + 2)
    print(f"\n{line}\n  {title}\n{line}", flush=True)

def step(msg: str):
    print(f"→ {msg}", flush=True)

def ok(msg: str):
    print(f"✅ {msg}", flush=True)

def warn(msg: str):
    print(f"⚠️  {msg}", flush=True)

def fail(msg: str):
    print(f"❌ {msg}", flush=True)

def run_cmd(cmd):
    print("   ▶ " + " ".join(cmd), flush=True)
    # Stream child output to our console so the user sees progress
    return subprocess.call(cmd)


# --------------------- Stage map & CLI ---------------------
STAGE_ORDER = [
    ("S0", "discover"),
    ("S1", "sampling"),
    ("S2", "preprocess"),
    ("S3", "angles"),
    ("S4", "features"),
    ("S5", "unify"),
    ("S6", "pca"),
    ("S7", "deltas"),
    ("S8", "paired"),
    ("S9", "lopo"),
]
CODE_BY_NAME = {code: idx for idx, (code, _) in enumerate(STAGE_ORDER)}
CODE_BY_LABEL = {label: idx for idx, (code, label) in enumerate(STAGE_ORDER)}

def parse_args():
    ap = argparse.ArgumentParser(description="Config-driven motion analysis pipeline with stage-resume.")
    ap.add_argument("start", nargs="?", default="S0",
                    help="Stage to start at. Use S0..S9 or names (discover, sampling, preprocess, angles, features, unify, pca, deltas, paired, lopo).")
    ap.add_argument("--start-at", dest="start_at", default=None,
                    help="Alternate way to pass start stage (same values as positional arg).")
    return ap.parse_args()

def normalize_start(value: str) -> int:
    v = (value or "S0").strip().lower()
    if v.upper() in CODE_BY_NAME:
        return CODE_BY_NAME[v.upper()]
    if v in CODE_BY_LABEL:
        return CODE_BY_LABEL[v]
    fail(f"Unknown start stage '{value}'. Valid: " +
         ", ".join([f"{c} ({n})" for c, n in STAGE_ORDER]))
    sys.exit(1)


# --------------------- Parse helpers ---------------------
def parse_participant(name: str):
    m = re.findall(r"\d+", str(name))
    return m[0] if m else None

def parse_session(name: str):
    s = str(name).lower()
    if "1st" in s: return "First"
    if "last" in s: return "Last"
    return "Unknown"


# --------------------- Main ---------------------
def main():
    t0 = time.perf_counter()
    args = parse_args()
    start_arg = args.start_at if args.start_at else args.start
    start_idx = normalize_start(start_arg)
    banner(f"Starting pipeline at: {STAGE_ORDER[start_idx][0]} ({STAGE_ORDER[start_idx][1]})")

    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        fail("config.yaml not found at repo root.")
        sys.exit(1)

    banner("Loading config.yaml")
    with open(cfg_path, "r") as f:
        C = yaml.safe_load(f)
    ok("Config loaded.")
    sys.stdout.flush()

    P   = C["paths"]
    S   = C["sampling"]
    PR  = C["preprocess"]
    FEA = C["features"]
    PCAc= C["pca"]
    DEL = C["deltas"]
    PST = C["paired_stats"]
    LOP = C["lopo"]
    PART= C["participants"]

    raw_dir         = Path(P["raw_dir"])
    data_clean      = Path(P["data_clean_dir"]);        data_clean.mkdir(parents=True, exist_ok=True)
    data_pp         = Path(P["data_preprocessed_dir"]); data_pp.mkdir(parents=True, exist_ok=True)
    data_angles     = Path(P["data_angles_dir"]);       data_angles.mkdir(parents=True, exist_ok=True)
    data_features   = Path(P["data_features_dir"]);     data_features.mkdir(parents=True, exist_ok=True)
    mdi_out         = Path(P["mdi_out_dir"]);           mdi_out.mkdir(parents=True, exist_ok=True)
    out_unified     = Path(P["out_unified"])

    # Placeholders that later stages will consume
    csvs = []
    cleaned = []
    pp_files = []
    angle_files = []
    feat_files = []
    mdi_table = None

    # ============= S0 — Discover raw CSVs =============
    if start_idx <= CODE_BY_LABEL["discover"]:
        banner("Stage S0 — Discover raw CSVs")
        if not raw_dir.exists():
            fail(f"Raw directory does not exist: {raw_dir}")
            sys.exit(1)
        csvs = sorted(raw_dir.glob("*.csv"))
        if not csvs:
            fail(f"No CSVs found in {raw_dir}")
            sys.exit(1)
        for f in csvs:
            step(f"Found: {f}")
        ok(f"Discovered {len(csvs)} file(s).")
    else:
        # When resuming, we may still need raw list (for names). Not strictly required past S1.
        csvs = sorted(raw_dir.glob("*.csv"))

    # ============= S1 — Sampling / resample =============
    if start_idx <= CODE_BY_LABEL["sampling"]:
        banner("Stage S1 — Sampling diagnostics & uniform time (check_sampling.py)")
        cleaned = []
        for idx, f in enumerate(csvs, 1):
            step(f"[{idx}/{len(csvs)}] Checking/resampling: {f.name}")
            rc = run_cmd([
                sys.executable, "check_sampling.py",
                "--in", str(f),
                "--outdir", str(data_clean),
                "--target-fps", str(S["target_fps"]),
                "--timebase", S["timebase"],
                "--method", S["method"],
                "--tolerance", str(S["tolerance"]),
                "--timescale", S["timescale"],
                "--tick-scale", str(S["tick_scale"])
            ] + (["--plots"] if bool(S.get("plots", False)) else []))
            if rc != 0:
                warn(f"check_sampling failed for {f.name}; skipping.")
                continue
            outs = sorted(data_clean.glob(f"{f.stem}_*_{int(S['target_fps'])}Hz.csv"))
            if outs:
                out = outs[-1]
                step(f"✓ Output: {out}")
                cleaned.append(out)
            else:
                warn(f"No cleaned output detected for {f.name}")
        if not cleaned:
            fail("No cleaned outputs produced; aborting.")
            sys.exit(1)
        ok(f"Sampling stage complete: {len(cleaned)} file(s).")
    else:
        # Resuming: collect already-resampled inputs
        cleaned = sorted(data_clean.glob("*_resampled_*Hz.csv"))
        if not cleaned:
            fail("No resampled files found in data_clean/. Cannot resume at or after preprocess.")
            sys.exit(1)
        banner("S1 skipped — using existing resampled files")
        for f in cleaned:
            step(f"{f.name}")
        ok(f"{len(cleaned)} cleaned file(s) available.")

    # ============= S2 — Preprocess =============
    if start_idx <= CODE_BY_LABEL["preprocess"]:
        banner("Stage S2 — View/size normalize + low-pass filter (preprocess.py)")
        for idx, c in enumerate(cleaned, 1):
            step(f"[{idx}/{len(cleaned)}] Preprocessing: {c.name}")
            args = [
                sys.executable, "preprocess.py",
                "--in", str(c),
                "--outdir", str(data_pp),
                "--fps", str(PR["fps"]),
                "--cutoff", str(PR["cutoff_hz"]),
                "--order", str(PR["order"])
            ]
            if PR.get("shoulder_override_m") is not None:
                args += ["--shoulder", str(PR["shoulder_override_m"])]
            else:
                args += [
                    "--participants", str(P["participants_table"]),
                    "--participant-col", PART["participant_col"],
                    "--shoulder-col", PART["shoulder_col"]
                ]
            rc = run_cmd(args)
            if rc != 0:
                warn(f"preprocess failed for {c.name}; skipping angles/features for this file.")
        pp_files = sorted(data_pp.glob("*_preprocessed.csv"))
        if not pp_files:
            fail("No preprocessed files found; aborting.")
            sys.exit(1)
        for ppf in pp_files:
            step(f"Preprocessed: {ppf.name}")
        ok(f"Preprocess stage complete: {len(pp_files)} file(s).")
    else:
        pp_files = sorted(data_pp.glob("*_preprocessed.csv"))
        if not pp_files:
            fail("No preprocessed files found in data_preprocessed/. Cannot resume at or after angles.")
            sys.exit(1)
        banner("S2 skipped — using existing preprocessed files")
        for f in pp_files:
            step(f"{f.name}")
        ok(f"{len(pp_files)} preprocessed file(s) available.")

    # ============= S3 — Joint angles =============
    if start_idx <= CODE_BY_LABEL["angles"]:
        banner("Stage S3 — Joint angles & velocities (joint_angles.py)")
        angle_files = []
        for idx, ppf in enumerate(pp_files, 1):
            step(f"[{idx}/{len(pp_files)}] Angles from: {ppf.name}")
            # FIX: call joint_angles.py with --in (no stdin). Also pass --outdir if your script supports it.
            args = [sys.executable, "joint_angles.py", "--in", str(ppf)]
            # If your joint_angles.py supports --outdir, uncomment the next line:
            args += ["--outdir", str(data_angles)]
            rc = run_cmd(args)
            if rc != 0:
                warn(f"joint_angles.py failed for {ppf.name}")
                continue
            # Prefer canonical output path; otherwise find any matching *_angles.csv
            out = data_angles / f"{ppf.stem}_angles.csv"
            if out.exists():
                step(f"✓ Output: {out.name}")
                angle_files.append(out)
            else:
                matches = sorted(data_angles.glob(f"{ppf.stem}*_angles.csv"))
                if matches:
                    step(f"✓ Output (found): {matches[-1].name}")
                    angle_files.append(matches[-1])
                else:
                    warn(f"No angles output detected for {ppf.name}")
        if not angle_files:
            fail("No angle files produced; aborting.")
            sys.exit(1)
        ok(f"Angles stage complete: {len(angle_files)} file(s).")
    else:
        angle_files = sorted(data_angles.glob("*_angles.csv"))
        if not angle_files:
            fail("No angles found in data_angles/. Cannot resume at or after features.")
            sys.exit(1)
        banner("S3 skipped — using existing angle files")
        for f in angle_files:
            step(f"{f.name}")
        ok(f"{len(angle_files)} angle file(s) available.")

    # ============= S4 — Features =============
    if start_idx <= CODE_BY_LABEL["features"]:
        banner("Stage S4 — Feature extraction (features.py)")
        feat_files = []
        for idx, a in enumerate(angle_files, 1):
            step(f"[{idx}/{len(angle_files)}] Features from: {a.name}")
            args = [
                sys.executable, "features.py",
                "--in", str(a),
                "--outdir", str(data_features),
                "--vel-max", str(FEA["vel_max"]),
                "--spike-frames", str(FEA["spike_frames_unmask"]),
                "--active-threshold", str(FEA["active_threshold"]),
            ]
            if FEA.get("write_long"): args += ["--long"]
            if FEA.get("trust_input"): args += ["--trust-input"]
            rc = run_cmd(args)
            if rc != 0:
                warn(f"features failed for {a.name}")
                continue
            out = data_features / f"{a.stem}_features.csv"
            if out.exists():
                step(f"✓ Output: {out.name}")
                feat_files.append(out)
        if not feat_files:
            fail("No features files produced; aborting.")
            sys.exit(1)
        ok(f"Features stage complete: {len(feat_files)} file(s).")
    else:
        feat_files = sorted(data_features.glob("*_features.csv"))
        if not feat_files:
            fail("No features found in data_features/. Cannot resume at or after unify.")
            sys.exit(1)
        banner("S4 skipped — using existing features files")
        for f in feat_files:
            step(f"{f.name}")
        ok(f"{len(feat_files)} feature file(s) available.")

    # ============= S5 — Unify =============
    if start_idx <= CODE_BY_LABEL["unify"]:
        banner("Stage S5 — Unify features")
        rows = []
        for ff in feat_files:
            step(f"Reading: {ff.name}")
            df = pd.read_csv(ff)
            df.insert(0, "CSV_name", ff.name.replace("_features.csv", ""))
            df["Participant"] = df["CSV_name"].apply(parse_participant)
            df["Session"] = df["CSV_name"].apply(parse_session)
            rows.append(df)
        unified = pd.concat(rows, ignore_index=True)
        before = len(unified)
        unified = unified[unified["Session"].isin(["First", "Last"])].copy()
        after = len(unified)
        meta = ["CSV_name", "Participant", "Session"]
        cols = meta + [c for c in unified.columns if c not in meta]
        unified = unified[cols]
        out_unified.parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(out_unified, index=False)
        ok(f"Unified: {out_unified.resolve()}  (rows kept: {after}/{before}, cols: {len(unified.columns)})")
    else:
        if not out_unified.exists():
            fail("unified_features.csv not found. Cannot resume at or after PCA.")
            sys.exit(1)
        banner("S5 skipped — using existing unified_features.csv")
        step(str(out_unified.resolve()))
        ok("Unified features available.")

    # ============= S6 — PCA (headless) =============
    if start_idx <= CODE_BY_LABEL["pca"]:
        banner("Stage S6 — PCA / MDI (PCA.py)")
        rc = run_cmd([
            sys.executable, "PCA.py",
            "--in", str(out_unified),
            "--outdir", str(mdi_out),
            "--include", PCAc["include"],
            "--max-missing", str(PCAc["max_missing"]),
            "--corr", str(PCAc["corr_thresh"]),
            "--keep", PCAc["keep_preference"],
            "--k", str(PCAc["k"]),
            "--plot-pcs", PCAc["plot_pcs"],
            "--noninteractive"
        ] + (["--autoprune"] if PCAc.get("autoprune") else []))
        if rc != 0:
            fail("PCA.py failed; aborting.")
            sys.exit(1)
        ok(f"PCA artifacts in: {mdi_out.resolve()}")
    else:
        banner("S6 skipped — using existing PCA artifacts (mdi_out)")
        step(str(mdi_out.resolve()))
        ok("PCA artifacts assumed present.")

    # ============= S7 — Deltas =============
    if start_idx <= CODE_BY_LABEL["deltas"]:
        banner("Stage S7 — Compute Δ tables (compute_deltas.py)")
        rc = run_cmd([
            sys.executable, "compute_deltas.py",
            "--unified", str(out_unified),
            "--outdir", str(mdi_out),
            "--include", C["deltas"]["include"]
        ] + (["--plot"] if C["deltas"].get("make_heatmaps") else []))
        if rc != 0:
            fail("compute_deltas.py failed; aborting.")
            sys.exit(1)
        ok("Δ tables saved.")
    else:
        banner("S7 skipped — assuming Δ tables present in mdi_out")
        ok("Δ tables assumed present.")

    # ============= S8 — Paired stats =============
    if start_idx <= CODE_BY_LABEL["paired"]:
        banner("Stage S8 — Paired within-subject stats (paired_mdi_stats.py)")
        mdi_candidates = sorted(mdi_out.glob("mdi_deltas_PC1..PC*.csv"))
        if not mdi_candidates:
            fail("mdi_deltas_PC1..PCk.csv not found in mdi_out; aborting.")
            sys.exit(1)
        mdi_table = mdi_candidates[-1]
        step(f"Using MDI table: {mdi_table.name}")
        rc = run_cmd([
            sys.executable, "paired_mdi_stats.py",
            "--mdi", str(mdi_table),
            "--metric", PST["metric"],
            "--boot-n", str(PST["boot_n"]),
            "--boot-seed", str(PST["boot_seed"])
        ])
        if rc != 0:
            fail("paired_mdi_stats.py failed; aborting.")
            sys.exit(1)
        ok("Paired stats saved (mdi_within_stats.csv).")
    else:
        mdi_candidates = sorted(mdi_out.glob("mdi_deltas_PC1..PC*.csv"))
        mdi_table = mdi_candidates[-1] if mdi_candidates else None
        banner("S8 skipped — assuming paired stats previously computed")
        ok("Paired stats assumed present.")

    # ============= S9 — LOPO logistic =============
    if start_idx <= CODE_BY_LABEL["lopo"]:
        banner("Stage S9 — LOPO logistic (lopo_logistic.py)")
        if mdi_table is None:
            mdi_candidates = sorted(mdi_out.glob("mdi_deltas_PC1..PC*.csv"))
            if not mdi_candidates:
                fail("mdi_deltas_PC1..PCk.csv not found; cannot run LOPO.")
                sys.exit(1)
            mdi_table = mdi_candidates[-1]
        rc = run_cmd([
            sys.executable, "lopo_logistic.py",
            "--scores", str(mdi_out / "pca_scores.csv"),
            "--mdi", str(mdi_table),
            "--outdir", str(mdi_out),
            "--permutations", str(LOP["permutations"]),
            "--seed", str(LOP["seed"])
        ])
        if rc != 0:
            fail("lopo_logistic.py failed; aborting.")
            sys.exit(1)
        ok("LOPO results saved (lopo_logistic_results.csv).")
    else:
        banner("S9 skipped — assuming LOPO results present")
        ok("LOPO results assumed present.")

    # ---------------- Done ----------------
    elapsed = time.perf_counter() - t0
    banner("Pipeline complete")
    print(f"Artifacts folder : {mdi_out.resolve()}", flush=True)
    print(f"Unified features : {out_unified.resolve()}", flush=True)
    print(f"Total elapsed    : {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
