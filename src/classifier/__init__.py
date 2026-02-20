"""
Classifier Module - Genomic Classification & Cell Segmentation Helpers

This module provides tools for training and evaluating classifiers
on genomic feature vectors (e.g., PAM50 subtype classification), as well as
cell segmentation and classification for histopathology tiles.

Components:
    - train_genomic_linear_clf: Train a linear classifier on H5 features
    - evaluate_genomic_clf: Evaluate classifier and perform diagnostics
    - segment_and_classify_cells: Run DeepCMorph cell segmentation on H&E tiles
      to produce per-cell-type binary masks (.npy) compatible with TopoFD

Usage:
    python -m src.classifier.train_genomic_linear_clf --help
    python -m src.classifier.evaluate_genomic_clf --help
    python -m src.classifier.segment_and_classify_cells --help
"""
