# Master-Thesis-Repository

This repository contains code for my Master Thesis entitled "Reconstructing histological images from genomic data using diffusion models".

## Project Structure

```
Master-Thesis-Repository/
├── src/                           # Main source code
│   ├── joint_training/            # Joint Genomic VAE + Diffusion training
│   │   ├── train.py               # Training orchestration
│   │   ├── model.py               # JointLitModel implementation
│   │   ├── dataset.py             # GenomicTileDataset
│   │   └── JOINT_TRAINING_SUMMARY.md
│   ├── quality_assurance/         # Evaluation: metrics, segmentation, topological analysis
│   │   ├── evaluate_reconstruction.py
│   │   ├── topological_frechet_distance.py
│   │   ├── segment_and_compute_topofd.py
│   │   ├── metrics.py
│   │   ├── utils.py               # Shared utilities
│   │   └── EVALUATION_GUIDE.md
│   ├── encoding/                  # Genomic feature encoding, VAE training
│   ├── classifier/                # Cell segmentation (DeepCMorph)
│   ├── preprocessing/             # Data preparation utilities
│   ├── finetune_diffusion/        # Diffusion model fine-tuning (legacy)
│   ├── visualization/             # Plotting and visualization helpers
│   ├── statistics/                # Statistical analysis
│   └── config.yaml                # Central configuration file
│
├── notebooks/                     # Jupyter notebooks for exploration & analysis
│
├── slurm/                         # HPC job submission scripts
│
├── tests/                         # Unit tests
│
├── plots/                         # Generated visualizations & results
│
├── run_pipeline.py                # Main entry point for pipeline execution
│
├── CODE_CLEANUP_ANALYSIS.md       # Detailed code review and consolidation report
│
├── CLEANUP_SUMMARY.md             # Summary of recent refactoring
│
├── pyproject.toml                 # Project metadata & dependencies
│
└── README.md                      # This file
```

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

## Usage

## Third-Party Credits and Provenance

This repository reuses and builds on external projects. Credit is listed here and
also inline in relevant source modules.

- **mopadi** (KatherLab/mopadi):
	- Repository: https://github.com/KatherLab/mopadi
	- Used directly for diffusion training infrastructure (`LitModel`, `ema`,
		`TrainConfig`, template configs) in:
		- `src/joint_training/model.py`
		- `src/finetune_diffusion/finetune_diffusion_with_genomic.py`
	- Preprocessing utilities were adapted from mopadi data-prep scripts in:
		- `src/preprocessing/get_tiles_within_rois.py`
		- `src/preprocessing/utils.py`

- **TopoCellGen** (Melon-Xu/TopoCellGen):
	- Repository: https://github.com/Melon-Xu/TopoCellGen
	- Topological Fréchet Distance implementation in
		`src/quality_assurance/topological_frechet_distance.py` is inspired by
		TopoCellGen's `eval_TopoFD` approach.

- **multiomics-open-research/Bulk-RNA-Bert**:
	- Repository: https://github.com/multiomics-open-research/Bulk-RNA-Bert
	- The gene-token transformer direction is conceptually inspired by this line
		of work.

For questions or contributions, please open an issue or pull request!

