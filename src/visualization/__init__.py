"""
Visualization Module
====================

Reusable, Crameri-coloured plotting utilities for every stage of the
histopathology generation pipeline.

Sub-modules
-----------
core
    Shared styling, colour-map helpers, and figure save/show utilities.
embedding
    HDF5 embedding loading, clinical-data merging, matrix preparation.
latent_space
    UMAP, t-SNE, PCA projections, silhouette analysis, dendrograms,
    cosine-distance heatmaps, K-means confusion matrices.
distributions
    Count plots, stacked bars, mosaic plots for categorical data.
training_plots
    Loss curves, batch-loss trajectories, LR schedules, early-stopping
    plots, and multi-panel training summaries for diffusion fine-tuning.

Quick start
-----------
>>> from src.visualization import latent_space as lsv
>>> from src.visualization import embedding as emb
>>>
>>> df_emb = emb.collect_embeddings("/path/to/features")
>>> X = emb.build_embedding_matrix(df_emb)
>>> labels = emb.prepare_labels(df_emb, "split")
>>> X_umap, fig = lsv.plot_umap(X, labels, save_path="umap.png")

Colour maps
-----------
All plots default to Fabio Crameri's perceptually uniform scientific colour
maps (``cmcrameri``).  Override locally by passing ``cmap_name=...`` or
globally by setting ``core.CATEGORICAL_CMAP`` etc.
"""

# -- public API --------------------------------------------------------
from .core import (
    build_label_palette,
    get_categorical_colors,
    get_crameri_cmap,
    save_figure,
    setup_style,
    show_or_save,
)
from .distributions import (
    plot_countplot,
    plot_mosaic,
    plot_stacked_bar,
)
from .embedding import (
    build_embedding_matrix,
    collect_embeddings,
    load_embedding,
    prepare_labels,
    read_clinical_data,
)
from .latent_space import (
    compute_pca,
    compute_silhouette,
    compute_tsne,
    compute_umap,
    plot_composite_latent_analysis,
    plot_cosine_distance_clustermap,
    plot_dendrogram,
    plot_kmeans_confusion,
    plot_pca,
    plot_pca_variance,
    plot_projection,
    plot_silhouette_per_group,
    plot_tsne,
    plot_umap,
    plot_umap_interactive,
)
from .training_plots import (
    plot_batch_loss_trajectory,
    plot_early_stopping,
    plot_loss_curves,
    plot_lr_schedule,
    plot_run_comparison,
    plot_train_val_comparison,
    plot_training_summary,
)
from .tfd_separability import (
    plot_channel_contributions,
    plot_radar_channel_contributions,
    plot_cell_type_boxplots,
    plot_ternary_composition,
    plot_nn_distance_violins,
    plot_cross_type_proximity_heatmaps,
    run_tfd_separability_viz,
)

__all__ = [
    # core
    "setup_style",
    "get_crameri_cmap",
    "get_categorical_colors",
    "save_figure",
    "show_or_save",
    "build_label_palette",
    # embedding I/O
    "load_embedding",
    "collect_embeddings",
    "read_clinical_data",
    "build_embedding_matrix",
    "prepare_labels",
    # latent-space
    "compute_umap",
    "compute_tsne",
    "compute_pca",
    "compute_silhouette",
    "plot_projection",
    "plot_umap",
    "plot_tsne",
    "plot_pca",
    "plot_pca_variance",
    "plot_umap_interactive",
    "plot_cosine_distance_clustermap",
    "plot_silhouette_per_group",
    "plot_dendrogram",
    "plot_kmeans_confusion",
    "plot_composite_latent_analysis",
    # distributions
    "plot_countplot",
    "plot_stacked_bar",
    "plot_mosaic",
    # training plots
    "plot_loss_curves",
    "plot_batch_loss_trajectory",
    "plot_train_val_comparison",
    "plot_lr_schedule",
    "plot_early_stopping",
    "plot_training_summary",
    "plot_run_comparison",
    # tfd separability
    "plot_channel_contributions",
    "plot_radar_channel_contributions",
    "plot_cell_type_boxplots",
    "plot_ternary_composition",
    "plot_nn_distance_violins",
    "plot_cross_type_proximity_heatmaps",
    "run_tfd_separability_viz",
]
