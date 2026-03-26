# Basal vs LumA Classifier (Evaluation-Only)

## Overview
This module trains and evaluates a subtype classifier on Virchow2 tile features.
It is intended for post-hoc evaluation of diffusion outputs and is not used to train the diffusion model.

Current implementation:
- Binary task: Basal vs LumA
- Label source: Majority_Subtype_mRNA (patient-level, assigned to each tile)
- Prediction unit: tile-level primary, patient-level aggregated summary secondary
- Feature format: H5/HDF5 files (typically one file per patient)

## Implemented components
- [src/classifier/utils_subtype_data.py](src/classifier/utils_subtype_data.py)
  - Loads YAML config and split JSON
  - Canonicalizes patient IDs
  - Loads and filters subtype labels to Basal/LumA
  - Infers feature/coord datasets from H5 files
  - Builds tile-level table with split membership
- [src/classifier/train_subtype_classifier.py](src/classifier/train_subtype_classifier.py)
  - Trains logistic regression with train-only standardization
  - Tunes decision threshold on validation split (balanced accuracy)
  - Saves model artifact and training summary
- [src/classifier/evaluate_subtype_classifier.py](src/classifier/evaluate_subtype_classifier.py)
  - Evaluates on val/test splits
  - Computes tile metrics, patient-summary metrics, hard-tile robustness metrics
- [run_pipeline.py](run_pipeline.py)
  - Adds stage: subtype_classifier
  - Runs train/evaluate/both from config section

## Data and split policy
The classifier uses exactly the diffusion split logic:
- Patient-level split only (train/val/test from patient_splits.json)
- No patient leakage across splits
- Tiles inherit the split of their patient

Subtype filtering:
- Keep only rows with subtype Basal or LumA
- Drop all other subtypes and missing labels

## H5 feature handling
Expected input is a directory with .h5/.hdf5 files.

Dataset inference behavior:
- Searches all datasets recursively in each H5
- Picks best feature candidate by name and shape heuristics
- Optionally picks coord dataset when available and shape-compatible
- If patient ID is not present in H5 keys, infers from filename

Tile table schema produced internally:
- patient_id
- split
- subtype
- tile_index
- feature
- optional x, y coordinates

## Model and metrics
Model:
- LogisticRegression (linear classifier)

Preprocessing:
- StandardScaler fit on train split only
- Applied unchanged to val/test

Tile-level metrics:
- ROC-AUC
- PR-AUC
- Balanced accuracy
- Macro F1
- Class-wise precision/recall
- Confusion matrix (raw + normalized)

Patient-summary metrics:
- Mean probability aggregation per patient
- Patient-level ROC-AUC and balanced accuracy

Hard-tile robustness metrics:
- Confident tile fraction
- Bottom-quartile confidence accuracy
- Mean per-patient probability variance

## Config-driven execution
Configured via section subtype_classifier in [src/config.yaml](src/config.yaml).

Supported mode values:
- train
- evaluate
- both

Pipeline command:
- python run_pipeline.py --config src/config.yaml --stage subtype_classifier

When mode is both:
- Train output: <output_dir>/train
- Eval output: <output_dir>/eval
- Default model path for eval: <output_dir>/train/subtype_linear_model.joblib

## Direct script execution
Train:
- python -m src.classifier.train_subtype_classifier --config src/config.yaml --features-dir <path> --patient-splits-path <path> --output-dir <path>

Evaluate:
- python -m src.classifier.evaluate_subtype_classifier --config src/config.yaml --features-dir <path> --patient-splits-path <path> --model-path <path> --output-dir <path>

## Outputs
Training outputs:
- subtype_linear_model.joblib
- train_summary.json

Evaluation outputs:
- evaluation_metrics.json

## Notes for diffusion-output evaluation
For generated/reconstructed tiles, run the same feature extraction and classifier inference tile-by-tile, then compare predicted subtype behavior across:
- true conditioning
- swapped conditioning
- zero conditioning

Report both tile-level shifts and patient-aggregated shifts.
