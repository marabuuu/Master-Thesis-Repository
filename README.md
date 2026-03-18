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

For questions or contributions, please open an issue or pull request!

