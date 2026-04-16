# Master-Thesis-Repository

This repository contains code for my Master Thesis entitled "Reconstructing histological images from genomic data using diffusion models".

## Project Structure

```
Master-Thesis-Repository/
├── src/                                   # Main source code
│   ├── config.yaml                        # Central pipeline configuration
│   ├── preprocessing/                     # ROI/tile preparation and feature build helpers
│   ├── encoding/                          # Genomic encoding and latent extraction utilities
│   ├── joint_training/                    # Baseline joint genomic + diffusion training
│   ├── cross_attention_joint_training/    # Cross-attention joint variant
│   ├── gene_token_transformer_joint_training/
│   ├── gene_token_cross_attention_joint_training/
│   ├── reconstruction/                    # Image-guided / random-noise reconstruction
│   ├── quality_assurance/                 # Reconstruction metrics and TopoFD
│   ├── visualization/                     # Plotting utilities
│   ├── classifier/                        # Virchow2 + subtype classification + segmentation
│   ├── statistics/                        # Dataset/training-statistics analysis
│   ├── finetune_diffusion/                # Legacy fine-tuning utilities
│   └── mopadi_genomic/                    # MoPaDi-genomic training entrypoint
├── slurm/                                 # HPC job submission scripts
├── notebooks/                             # Analysis notebooks
├── tests/                                 # Unit tests
├── run_pipeline.py                        # Main orchestration CLI
├── pyproject.toml                         # Project metadata and dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.11+

### Installation

## Usage

## Building on the following Work

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


For questions or contributions, please open an issue or pull request!

