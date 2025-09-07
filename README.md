# Quantifying Motor Diversity in Gaga Dance Practice Using Machine Learning

A machine-learning pipeline for extracting kinematic features from motion-capture data, computing a Motor Diversity Index (MDI), and evaluating learning-related changes in Gaga dance practice.

---

## Project Overview
Motor diversity reflects the range and variability of human movement, yet conventional motor learning measures (e.g., speed, accuracy) often overlook this dimension. Gaga, a movement language designed to expand expression beyond habitual patterns, provides a unique context for studying motor variability. Establishing this framework in a control group sets the stage for future investigations into whether psilocybin—known to enhance creativity and disrupt habitual behaviors—can further amplify motor divers...

**Key contributions:**
- A reproducible pipeline for processing motion-capture data into standardized kinematic features.  
- The **Motor Diversity Index (MDI)**, a variance-weighted PCA-based metric for quantifying movement diversity.  
- Pilot evidence from *n* = 7 participants demonstrating increased diversity after one month of Gaga practice.  
- A methodological foundation for future studies combining Gaga with psilocybin interventions.  

---

## Pipeline Description
1. **Data Preparation (S1–S2):** Resampling (120 Hz), interpolation, body-fixed projection, normalization by shoulder width, and filtering (Butterworth + Savitzky–Golay).  
2. **Feature Engineering (S3–S4):** Extract RMS, STD, CV, ZeroCross per joint; compute global MeanActiveJoints.  
3. **Dimensionality Reduction (S5–S6):** PCA on standardized features; define variance-weighted MDI.  
4. **Learning Effects (S7–S8):** Compute session deltas; run paired $t$-tests, Wilcoxon signed-rank, and bootstrap CIs.  
5. **Generalization (S9):** LOPO logistic regression (PCs only, and PCs+MAJ); evaluate AUROC with permutation testing.  
6. **Visualization:** Generate six unified figures (scree plot, loadings, trajectories, feature deltas, MDI stats, ROC curve).  

---

## Repository Structure
```
.
├── data_preprocessed/       # Cleaned joint trajectories (S2 output)
├── data_angles/             # Computed joint angles (S3 output)
├── data_features/           # Extracted per-joint features (S4 output)
├── mdi_out/                 # PCA, deltas, stats, and figures
│   ├── pca_variance_explained.csv
│   ├── pca_loadings.csv
│   ├── pca_scores.csv
│   ├── deltas_CV.csv
│   ├── delta_MeanActiveJoints.csv
│   ├── mdi_within_stats.csv
│   ├── lopo_logistic_results.csv
│   ├── lopo_predictions.csv
│   └── figures/ (fig01–fig06)
├── config.yaml              # Parameters: participants, sessions, filtering
├── preprocess.py            # Resampling, interpolation, filtering (S1–S2)
├── joint_angles.py          # Compute joint angles/velocities (S3)
├── features.py              # Extract kinematic features (S4)
├── PCA.py                   # Run PCA, compute MDI (S5–S6)
├── compute_deltas.py        # Session-to-session feature deltas (S7)
├── paired_mdi_stats.py      # Statistical testing (S8)
├── lopo_logistic.py         # LOPO classification (S9, PCs only)
├── lopo_logistic_plus_meanactive.py  # LOPO with PCs + MAJ
└── make_figures_from_mdi_out.py      # Unified plotting script (6 figures)
```

---

## Input and Output
**Input:**
- Raw motion-capture CSVs with 3D marker positions.  
- Participant metadata (`participants_shoulder_width.csv`).  

**Outputs:**
- Preprocessed trajectories, joint angles, and kinematic features.  
- PCA results, feature deltas, MDI statistics, classification metrics.  
- Publication-ready figures in `mdi_out/figures/`.  

---

## Installation
Requires Python ≥ 3.9. Install dependencies with:  
```bash
pip install -r requirements.txt
```  

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`.  

---

## Usage
Run the full pipeline step by step:  
```bash
# Preprocessing (resampling, filtering, normalization)
python preprocess.py --input data_raw/ --output data_preprocessed/

# Compute joint angles and velocities
python joint_angles.py --input data_preprocessed/ --output data_angles/

# Extract kinematic features
python features.py --input data_angles/ --output data_features/

# Run PCA and compute MDI
python PCA.py --input data_features/ --output mdi_out/

# Compute deltas and stats
python compute_deltas.py --input mdi_out/
python paired_mdi_stats.py --input mdi_out/

# Classification
python lopo_logistic.py --scores mdi_out/pca_scores.csv --unified mdi_out/unified_features.csv
python lopo_logistic_plus_meanactive.py --scores mdi_out/pca_scores.csv --unified mdi_out/unified_features.csv

# Generate unified figures
python make_figures_from_mdi_out.py --mdi-out mdi_out --outdir figures/
```  

---

## Results (Pilot Study, *n* = 7)
- First three PCs captured $\sim$77\% variance (PC1 = 36.3\%, PC2 = 29.7\%, PC3 = 11.0\%).  
- Participants showed consistent positive displacement along PC2, reflecting increased postural variability.  
- $\Delta$CV values were largely positive; MeanActiveJoints increased in 6/7 participants.  
- MDI increased significantly (mean $\Delta$ = 3.09, SD = 1.34; $t(6)=6.1$, $p<0.001$; Cohen’s $d=2.3$; bootstrap 95\% CI [2.21, 4.03]).  
- LOPO logistic regression (PCs only) achieved AUROC $pprox$ 0.78 ($p=0.012$ permutation).  
- Adding MAJ yielded AUROC = 1.0, but without permutation support ($p pprox 0.14$), consistent with overfitting.  

---

## Limitations and Future Directions
- Small sample size ($n=7$) and high feature-to-sample ratio limit generalizability and may inflate effect sizes.  
- Future work should include larger cohorts, develop additional kinematic features aligned with motor diversity, and explore alternative models such as mixed-effects, gradient boosting, time-series methods, or functional PCA.  
- Rigorous evaluation with nested cross-validation and resampling will be important to ensure robust and reproducible findings.  

---

## Citation
If you use this code or data, please cite:  

Hazan, D. (2025). *Quantifying Motor Diversity in Gaga Dance Practice Using Machine Learning Models*. Tel Aviv University, Final Project in Computational Models of Learning.
