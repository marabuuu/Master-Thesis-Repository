#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified CLI for running all pipeline stages from config.yaml

Usage:
    python run_pipeline.py --config src/config.yaml --stage evaluation
    python run_pipeline.py --config src/config.yaml --stage preprocessing
    python run_pipeline.py --config src/config.yaml --stage all
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure the local mopadi package is on sys.path when running from repo root.
# This avoids `No module named 'mopadi.configs'` when mopadi is available as sibling package.
repo_root = Path(__file__).resolve().parent
mopadi_src = (repo_root.parent / "mopadi" / "src").resolve()
if mopadi_src.exists() and str(mopadi_src) not in sys.path:
    sys.path.insert(0, str(mopadi_src))


def resolve_config_paths(config_dict: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """
    Recursively resolve relative paths in config relative to repo_root.
    
    Converts paths like "./data/..." or "experiments/..." to absolute paths
    based on the repository root. Leaves absolute paths unchanged.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary (may contain nested dicts and lists)
    repo_root : Path
        Repository root directory to use as base for relative paths
    
    Returns
    -------
    Dict[str, Any]
        Configuration with resolved paths
    """
    def _resolve_path(value: str) -> str:
        repo_candidate = (repo_root / value).resolve()
        normalized = value[2:] if value.startswith("./") else value

        # In this workspace layout, `data/`, `dataframes/`, and `experiments/`
        # are siblings of the repository root.
        # of the repository root. Use parent fallback when needed.
        if normalized.startswith(("data/", "dataframes/", "experiments/")):
            parent_candidate = (repo_root.parent / normalized).resolve()
            if parent_candidate.exists() or not repo_candidate.exists():
                return str(parent_candidate)

        return str(repo_candidate)

    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                resolve_config_paths(value, repo_root)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        resolve_config_paths(item, repo_root)
            elif isinstance(value, str):
                # Detect if this looks like a path:
                # - starts with ./ or ../
                # - contains path separators and common dir names, or is a relative path
                # - is NOT already absolute
                if not value.startswith('/') and (
                    value.startswith('./') or 
                    value.startswith('../') or
                    any(part in value for part in ['data/', 'experiments/', 'dataframes/', 'slurm/', 'src/'])
                ):
                    config_dict[key] = _resolve_path(value)
    
    return config_dict


def load_config(config_path: str, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and resolve relative paths.
    
    Parameters
    ----------
    config_path : str
        Path to config.yaml file
    repo_root : Path, optional
        Repository root for resolving relative paths. If None, inferred from config_path.
    
    Returns
    -------
    Dict[str, Any]
        Configuration with resolved paths
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if repo_root is None:
        # Infer repo root from config path: if config is "src/config.yaml", repo is parent
        config_path_obj = Path(config_path).resolve()
        repo_root = config_path_obj.parent if config_path_obj.name == "config.yaml" else config_path_obj
        if not (repo_root / "run_pipeline.py").exists():
            # Try parent if not found
            repo_root = repo_root.parent
    
    resolve_config_paths(config, repo_root)
    return config


def run_stage(config: Dict[str, Any], stage: str, config_path: str = "", verbose: bool = True) -> None:
    """Run a specific pipeline stage based on config."""
    
    if stage == "downscale_tiles":
        from src.preprocessing.downscale_tiles import run_downscale_tiles
        if "downscale_tiles" not in config:
            raise ValueError("No 'downscale_tiles' section in config.yaml")
        run_downscale_tiles(config["downscale_tiles"], verbose=verbose)

    elif stage == "evaluation":
        from src.quality_assurance import run_evaluation
        if "evaluation" not in config:
            raise ValueError("No 'evaluation' section in config.yaml")
        run_evaluation(config["evaluation"], verbose=verbose)
    
    elif stage == "preprocessing":
        from src.preprocessing.get_tiles_within_rois import main as run_roi_filter
        if "preprocessing" not in config:
            raise ValueError("No 'preprocessing' section in config.yaml")
        pre_cfg = config["preprocessing"]

        # Build sys.argv so argparse inside get_tiles_within_rois.main() picks up
        # the config path — all actual values are read from the YAML.
        import sys as _sys
        _sys.argv = ["get_tiles_within_rois", "--config", config_path]
        run_roi_filter()

    elif stage == "tar_to_tumor_zip":
        from src.preprocessing.tar_to_tumor_zip import main as run_tar_to_tumor_zip
        if "tar_to_tumor_zip" not in config:
            raise ValueError("No 'tar_to_tumor_zip' section in config.yaml")
        import sys as _sys
        _sys.argv = ["tar_to_tumor_zip", "--config", config_path]
        run_tar_to_tumor_zip()
    
    elif stage == "encoding":
        print(f"[INFO] Encoding stage not yet configured in this CLI")
        print(f"       Run: python -m src.encoding.encode_genomics --config {config_path}")
    
    elif stage == "visualize_latents":
        from src.visualization.visualize_latents import run_visualizations
        if "visualize_latents" not in config:
            raise ValueError("No 'visualize_latents' section in config.yaml")
        run_visualizations(config["visualize_latents"], verbose=verbose)

    elif stage == "poc_breast_vs_liver_visualize_latents":
        from src.visualization.visualize_latents import run_visualizations
        if "poc_breast_vs_liver_visualize_latents" not in config:
            raise ValueError("No 'poc_breast_vs_liver_visualize_latents' section in config.yaml")
        run_visualizations(config["poc_breast_vs_liver_visualize_latents"], verbose=verbose)
    
    elif stage == "training":
        print(f"[INFO] Training stage not yet configured in this CLI")
        print(f"       Run: python -m src.training.train_genomic_autoenc --config {config_path}")
    
    elif stage == "dataset_statistics":
        from src.statistics.dataset_statistics import run_dataset_statistics
        if "dataset_statistics" not in config:
            raise ValueError("No 'dataset_statistics' section in config.yaml")
        run_dataset_statistics(config["dataset_statistics"], verbose=verbose)

    elif stage == "training_stats":
        # This stage wraps src.statistics.training_curves CLI using the provided config
        from src.statistics import training_curves
        if "training_stats" not in config:
            raise ValueError("No 'training_stats' section in config.yaml")
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], "--config", config_path]
            training_curves.main()
        finally:
            sys.argv = old_argv

    elif stage == "sampling":
        print(f"[INFO] Sampling stage not yet configured in this CLI")
        print(f"       Run: python -m src.sampling.sample_from_model --config {config_path}")
    
    elif stage == "reconstruction":
        from src.reconstruction import run_reconstruction
        if "reconstruction" not in config:
            raise ValueError("No 'reconstruction' section in config.yaml")
        run_reconstruction(config["reconstruction"], verbose=verbose)
    
    elif stage == "segmentation":
        from src.classifier.segment_and_classify_cells import run_segmentation
        if "segmentation" not in config:
            raise ValueError("No 'segmentation' section in config.yaml")
        run_segmentation(config["segmentation"], verbose=verbose)

    elif stage == "tfd_separability":
        from src.quality_assurance.tfd_separability import run_tfd_separability
        if "tfd_separability" not in config:
            raise ValueError("No 'tfd_separability' section in config.yaml")
        run_tfd_separability(config["tfd_separability"], verbose=verbose)

    elif stage == "tfd_separability_poc":
        from src.quality_assurance.tfd_separability import run_tfd_separability
        if "tfd_separability_poc" not in config:
            raise ValueError("No 'tfd_separability_poc' section in config.yaml")
        run_tfd_separability(config["tfd_separability_poc"], verbose=verbose)

    elif stage == "tfd_separability_viz":
        from src.visualization.tfd_separability import run_tfd_separability_viz
        if "tfd_separability_viz" not in config:
            raise ValueError("No 'tfd_separability_viz' section in config.yaml")
        run_tfd_separability_viz(config["tfd_separability_viz"], verbose=verbose)

    elif stage == "tfd_separability_generated":
        from src.quality_assurance.tfd_separability import run_tfd_separability_from_dirs
        if "tfd_separability_generated" not in config:
            raise ValueError("No 'tfd_separability_generated' section in config.yaml")
        run_tfd_separability_from_dirs(config["tfd_separability_generated"], verbose=verbose)

    elif stage == "virchow2_extraction":
        from src.classifier import run_virchow2_extraction
        if "virchow2_extraction" not in config:
            raise ValueError("No 'virchow2_extraction' section in config.yaml")
        run_virchow2_extraction(dict(config["virchow2_extraction"]), verbose=verbose)

    elif stage == "subtype_classifier":
        from src.classifier import run_subtype_classifier
        if "subtype_classifier" not in config:
            raise ValueError("No 'subtype_classifier' section in config.yaml")
        cfg = dict(config["subtype_classifier"])
        cfg.setdefault("config_path", config_path)
        run_subtype_classifier(cfg, verbose=verbose)

    elif stage == "build_genomic_features":
        from src.preprocessing.build_genomic_features import run_build_genomic_features
        if "build_genomic_features" not in config:
            raise ValueError("No 'build_genomic_features' section in config.yaml")
        run_build_genomic_features(config["build_genomic_features"], verbose=verbose)

    elif stage == "poc_breast_vs_liver_genomic_features":
        from src.preprocessing.build_genomic_features import run_build_genomic_features
        if "poc_breast_vs_liver_genomic_features" not in config:
            raise ValueError("No 'poc_breast_vs_liver_genomic_features' section in config.yaml")
        run_build_genomic_features(config["poc_breast_vs_liver_genomic_features"], verbose=verbose)

    elif stage == "genomic_adapter_training":
        from src.drafts.genomic_adapter.run_training import run_gda_training
        if "genomic_adapter_training" not in config:
            raise ValueError("No 'genomic_adapter_training' section in config.yaml")
        run_gda_training(config["genomic_adapter_training"], verbose=verbose)

    elif stage == "poc_breast_vs_liver_gda":
        from src.drafts.genomic_adapter.run_training import run_gda_training
        if "poc_breast_vs_liver_gda" not in config:
            raise ValueError("No 'poc_breast_vs_liver_gda' section in config.yaml")
        run_gda_training(config["poc_breast_vs_liver_gda"], verbose=verbose)

    elif stage == "poc_breast_vs_liver_cfg":
        from src.drafts.genomic_adapter.run_training import run_gda_training
        if "poc_breast_vs_liver_cfg" not in config:
            raise ValueError("No 'poc_breast_vs_liver_cfg' section in config.yaml")
        run_gda_training(config["poc_breast_vs_liver_cfg"], verbose=verbose)

    elif stage == "poc_breast_vs_liver_cfg_brca_init":
        from src.drafts.genomic_adapter.run_training import run_gda_training
        if "poc_breast_vs_liver_cfg_brca_init" not in config:
            raise ValueError("No 'poc_breast_vs_liver_cfg_brca_init' section in config.yaml")
        run_gda_training(config["poc_breast_vs_liver_cfg_brca_init"], verbose=verbose)

    elif stage == "brca_pam50_cfg":
        from src.drafts.genomic_adapter.run_training import run_gda_training
        if "brca_pam50_cfg" not in config:
            raise ValueError("No 'brca_pam50_cfg' section in config.yaml")
        run_gda_training(config["brca_pam50_cfg"], verbose=verbose)

    elif stage == "brca_pam50_cfg_v2":
        from src.poc_experiment.run_cfg_training import run_cfg_training
        if "brca_pam50_cfg_v2" not in config:
            raise ValueError("No 'brca_pam50_cfg_v2' section in config.yaml")
        run_cfg_training(config["brca_pam50_cfg_v2"], verbose=verbose)

    elif stage == "poc_brca_lihc_cfg_v2":
        from src.poc_experiment.run_cfg_training import run_cfg_training
        if "poc_brca_lihc_cfg_v2" not in config:
            raise ValueError("No 'poc_brca_lihc_cfg_v2' section in config.yaml")
        run_cfg_training(config["poc_brca_lihc_cfg_v2"], verbose=verbose)

    elif stage == "poc_brca_lihc_cfg_v2_dgx":
        from src.poc_experiment.run_cfg_training import run_cfg_training
        if "poc_brca_lihc_cfg_v2_dgx" not in config:
            raise ValueError("No 'poc_brca_lihc_cfg_v2_dgx' section in config.yaml")
        run_cfg_training(config["poc_brca_lihc_cfg_v2_dgx"], verbose=verbose)

    elif stage in ("poc_128_1hot", "poc_128_1hot_nonorm", "poc_128_1hot_nonorm_30M", "poc_128_zero", "poc_128_zero_30M", "poc_128_noise", "poc_128_noise_30M", "poc_128_rna", "poc_128_rna_30M", "poc_128_class_embed_30M"):
        from src.poc_experiment.run_cfg_training import run_cfg_training
        if stage not in config:
            raise ValueError(f"No '{stage}' section in config.yaml")
        run_cfg_training(config[stage], verbose=verbose)

    elif stage == "virchow2_umap":
        from src.visualization.virchow2_umap import run_virchow2_umap
        if "virchow2_umap" not in config:
            raise ValueError("No 'virchow2_umap' section in config.yaml")
        run_virchow2_umap(config["virchow2_umap"], verbose=verbose)

    elif stage == "virchow2_umap_cohort":
        from src.visualization.virchow2_umap import run_virchow2_umap
        if "virchow2_umap_cohort" not in config:
            raise ValueError("No 'virchow2_umap_cohort' section in config.yaml")
        run_virchow2_umap(config["virchow2_umap_cohort"], verbose=verbose)

    elif stage == "all":
        print("[INFO] Running all stages in sequence...")
        for s in ["preprocessing", "encoding", "training", "sampling", "evaluation"]:
            print(f"\n{'='*70}")
            print(f"STAGE: {s.upper()}")
            print(f"{'='*70}\n")
            try:
                run_stage(config, s, config_path, verbose=verbose)
            except Exception as e:
                if "not yet configured" not in str(e):
                    print(f"[ERROR] Stage '{s}' failed: {e}")
                    return
    
    else:
        raise ValueError(
            f"Unknown stage: {stage}. Choose from: downscale_tiles, preprocessing, tar_to_tumor_zip, encoding, visualize_latents, poc_breast_vs_liver_visualize_latents, training, dataset_statistics, training_stats, sampling, reconstruction, segmentation, tfd_separability, tfd_separability_poc, tfd_separability_viz, tfd_separability_generated, subtype_classifier, build_genomic_features, poc_breast_vs_liver_genomic_features, genomic_adapter_training, poc_breast_vs_liver_gda, poc_breast_vs_liver_cfg, poc_breast_vs_liver_cfg_brca_init, brca_pam50_cfg, brca_pam50_cfg_v2, poc_brca_lihc_cfg_v2, virchow2_umap, evaluation, all"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for running pipeline stages from config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation only
  python run_pipeline.py --config src/config.yaml --stage evaluation
  
  # Run all stages
  python run_pipeline.py --config src/config.yaml --stage all
  
  # With quiet output
  python run_pipeline.py --config src/config.yaml --stage evaluation --quiet
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file (e.g., src/config.yaml)",
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["downscale_tiles", "preprocessing", "encoding", "visualize_latents", "poc_breast_vs_liver_visualize_latents", "training", "dataset_statistics", "training_stats", "sampling", "reconstruction", "segmentation", "tfd_separability", "tfd_separability_poc", "tfd_separability_viz", "tfd_separability_generated", "subtype_classifier", "build_genomic_features", "poc_breast_vs_liver_genomic_features", "genomic_adapter_training", "poc_breast_vs_liver_gda", "poc_breast_vs_liver_cfg", "poc_breast_vs_liver_cfg_brca_init", "brca_pam50_cfg", "brca_pam50_cfg_v2", "poc_brca_lihc_cfg_v2", "poc_brca_lihc_cfg_v2_dgx", "poc_128_1hot", "poc_128_1hot_nonorm", "poc_128_1hot_nonorm_30M", "poc_128_zero", "poc_128_zero_30M", "poc_128_noise", "poc_128_noise_30M", "poc_128_rna", "poc_128_rna_30M", "poc_128_class_embed_30M", "virchow2_umap", "virchow2_umap_cohort", "evaluation", "all"],
        help="Pipeline stage to run",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    try:
        # Determine repo root: directory containing this run_pipeline.py script
        repo_root = Path(__file__).resolve().parent
        
        # Load config and resolve all relative paths
        config = load_config(args.config, repo_root=repo_root)
        run_stage(config, args.stage, args.config, verbose=not args.quiet)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
