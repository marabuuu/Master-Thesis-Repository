"""
Latent-Space Visualisation
==========================

UMAP, t-SNE, PCA projections, silhouette analysis, hierarchical clustering,
and cosine-distance heatmaps – all styled with Crameri scientific colour maps.

Every public function follows the same convention:

* accepts a pre-built 2-D ``X`` matrix + label arrays,
* returns the matplotlib ``Figure`` **and** the transformed coordinates
  (where applicable) so callers can store them in a DataFrame,
* optionally saves / shows via the ``save_path`` / ``show`` arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .core import (
    CATEGORICAL_CMAP,
    DIVERGING_CMAP,
    HEATMAP_CMAP,
    SEQUENTIAL_CMAP,
    _check_matplotlib,
    _check_seaborn,
    get_categorical_colors,
    get_crameri_cmap,
    save_figure,
    setup_style,
    show_or_save,
)

# ── guarded imports ───────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    pass

try:
    import seaborn as sns
except ImportError:
    pass


# ===================================================================
#  Internal helpers
# ===================================================================


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    """Guarantee *X* is 2-D (samples × features)."""
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _unwrap_embedding(raw: Any) -> np.ndarray:
    """Handle tuple / sparse / ndarray returns from UMAP / t-SNE."""
    candidate = raw[0] if isinstance(raw, tuple) else raw
    if hasattr(candidate, "toarray"):
        return candidate.toarray()  # type: ignore[union-attr]
    return np.asarray(candidate)


def _build_palette(
    labels: np.ndarray,
    cmap_name: str = CATEGORICAL_CMAP,
) -> Dict[str, str]:
    """Map unique label strings to Crameri hex colours."""
    unique = sorted(np.unique(labels))
    colours = get_categorical_colors(len(unique), cmap_name=cmap_name)
    return dict(zip(unique, colours))


# ===================================================================
#  Dimensionality reduction
# ===================================================================


def compute_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
    **umap_kw: Any,
) -> np.ndarray:
    """Run UMAP and return the low-dimensional embedding.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    n_neighbors, min_dist, n_components, metric, random_state
        Standard UMAP hyper-parameters.
    **umap_kw
        Extra keyword arguments forwarded to ``umap.UMAP``.

    Returns
    -------
    np.ndarray, shape (n, n_components)
    """
    import umap as umap_lib

    X = _ensure_2d(X)
    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
        **umap_kw,
    )
    return _unwrap_embedding(reducer.fit_transform(X))


def compute_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30,
    learning_rate: float = 200,
    n_iter: int = 1000,
    metric: str = "cosine",
    random_state: int = 42,
    **tsne_kw: Any,
) -> np.ndarray:
    """Run t-SNE and return the low-dimensional embedding.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    n_components, perplexity, learning_rate, n_iter, metric, random_state
        Standard ``sklearn.manifold.TSNE`` hyper-parameters.
    **tsne_kw
        Extra keyword arguments forwarded to ``TSNE``.

    Returns
    -------
    np.ndarray, shape (n, n_components)
    """
    from sklearn.manifold import TSNE

    X = _ensure_2d(X)
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        init="random",
        **tsne_kw,
    )
    return _unwrap_embedding(reducer.fit_transform(X))


def compute_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    **pca_kw: Any,
) -> Tuple[np.ndarray, Any]:
    """Run PCA and return (transformed data, fitted PCA object).

    The caller can inspect ``pca.explained_variance_ratio_`` etc.

    Returns
    -------
    (np.ndarray, sklearn.decomposition.PCA)
    """
    from sklearn.decomposition import PCA

    X = _ensure_2d(X)
    pca = PCA(n_components=n_components, random_state=random_state, **pca_kw)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


# ===================================================================
#  Scatter plots for 2-D projections
# ===================================================================


def plot_projection(
    coords: np.ndarray,
    hue_labels: np.ndarray,
    style_labels: Optional[np.ndarray] = None,
    title: str = "2-D projection",
    axis_labels: Tuple[str, str] = ("Dim 1", "Dim 2"),
    cmap_name: str = CATEGORICAL_CMAP,
    markers: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (8, 6),
    point_size: int = 80,
    alpha: float = 0.8,
    edgecolor: str = "k",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **scatter_kw: Any,
) -> Figure:
    """Generic scatter plot for any 2-D projection (UMAP / t-SNE / PCA).

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
    hue_labels : ndarray
        Categorical labels used for colour.
    style_labels : ndarray, optional
        Categorical labels used for marker style (e.g. train / test).
    title, axis_labels : str / tuple
    cmap_name : str
        Crameri categorical colour map name.
    markers : dict, optional
        ``{label: marker_char}`` mapping for *style_labels*.
    figsize, point_size, alpha, edgecolor
        Visual tweaks.
    save_path : str | Path | None
    show : bool
    **scatter_kw
        Extra kwargs forwarded to ``sns.scatterplot``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_seaborn()
    setup_style()

    palette = _build_palette(hue_labels, cmap_name)

    tmp = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "hue": hue_labels}
    )
    if style_labels is not None:
        tmp["style"] = style_labels

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=tmp,
        x="x",
        y="y",
        hue="hue",
        style="style" if style_labels is not None else None,
        palette=palette,
        markers=markers,
        s=point_size,
        alpha=alpha,
        edgecolor=edgecolor,
        ax=ax,
        **scatter_kw,
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig


def plot_umap(
    X: np.ndarray,
    hue_labels: np.ndarray,
    style_labels: Optional[np.ndarray] = None,
    title: str = "UMAP of VAE embeddings",
    cmap_name: str = CATEGORICAL_CMAP,
    markers: Optional[Dict[str, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    umap_kw: Optional[Dict[str, Any]] = None,
    **plot_kw: Any,
) -> Tuple[np.ndarray, Figure]:
    """Compute UMAP + plot in one call.

    Returns
    -------
    (X_umap, fig) – the 2-D coordinates and the figure.
    """
    X_umap = compute_umap(X, **(umap_kw or {}))
    fig = plot_projection(
        X_umap,
        hue_labels,
        style_labels=style_labels,
        title=title,
        axis_labels=("UMAP1", "UMAP2"),
        cmap_name=cmap_name,
        markers=markers,
        save_path=save_path,
        show=show,
        **plot_kw,
    )
    return X_umap, fig


def plot_tsne(
    X: np.ndarray,
    hue_labels: np.ndarray,
    style_labels: Optional[np.ndarray] = None,
    title: str = "t-SNE of VAE embeddings",
    cmap_name: str = CATEGORICAL_CMAP,
    markers: Optional[Dict[str, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    tsne_kw: Optional[Dict[str, Any]] = None,
    **plot_kw: Any,
) -> Tuple[np.ndarray, Figure]:
    """Compute t-SNE + plot in one call.

    Returns
    -------
    (X_tsne, fig)
    """
    X_tsne = compute_tsne(X, **(tsne_kw or {}))
    fig = plot_projection(
        X_tsne,
        hue_labels,
        style_labels=style_labels,
        title=title,
        axis_labels=("tSNE1", "tSNE2"),
        cmap_name=cmap_name,
        markers=markers,
        save_path=save_path,
        show=show,
        **plot_kw,
    )
    return X_tsne, fig


def plot_pca(
    X: np.ndarray,
    hue_labels: np.ndarray,
    style_labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    cmap_name: str = CATEGORICAL_CMAP,
    markers: Optional[Dict[str, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    pca_kw: Optional[Dict[str, Any]] = None,
    **plot_kw: Any,
) -> Tuple[np.ndarray, Any, Figure]:
    """Compute PCA + plot in one call.

    Returns
    -------
    (X_pca, pca_object, fig)
    """
    X_pca, pca = compute_pca(X, **(pca_kw or {}))
    var = pca.explained_variance_ratio_.sum()
    if title is None:
        title = f"PCA (explained variance: {var:.2%})"
    fig = plot_projection(
        X_pca,
        hue_labels,
        style_labels=style_labels,
        title=title,
        axis_labels=("PC1", "PC2"),
        cmap_name=cmap_name,
        markers=markers,
        save_path=save_path,
        show=show,
        **plot_kw,
    )
    return X_pca, pca, fig


# ===================================================================
#  Interactive Plotly UMAP
# ===================================================================


def plot_umap_interactive(
    df: pd.DataFrame,
    umap1_col: str = "UMAP1",
    umap2_col: str = "UMAP2",
    color_col: str = "Majority_Subtype_mRNA",
    symbol_col: Optional[str] = "split",
    hover_cols: Optional[List[str]] = None,
    title: str = "Interactive UMAP – hover for details",
    cmap_name: str = CATEGORICAL_CMAP,
    width: int = 800,
    height: int = 600,
    save_html: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Any:
    """Create an interactive Plotly scatter from pre-computed UMAP coords.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *umap1_col*, *umap2_col*, *color_col*.
    save_html : str | Path | None
        If given, write an interactive HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    unique_labels = sorted(df[color_col].dropna().unique())
    colors = get_categorical_colors(len(unique_labels), cmap_name=cmap_name)

    fig = px.scatter(
        df,
        x=umap1_col,
        y=umap2_col,
        color=color_col,
        symbol=symbol_col,
        hover_data=hover_cols,
        title=title,
        width=width,
        height=height,
        color_discrete_sequence=colors,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="black")))
    if show:
        fig.show()
    if save_html is not None:
        save_html = Path(save_html)
        save_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))
        print(f"[OK] Saved interactive plot → {save_html}")
    return fig


# ===================================================================
#  Distance / similarity heat-maps
# ===================================================================


def plot_cosine_distance_clustermap(
    X: np.ndarray,
    n_samples: int = 200,
    cmap_name: str = HEATMAP_CMAP,
    figsize: Tuple[float, float] = (9, 9),
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Hierarchically-clustered cosine-distance heatmap on a random subset.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    n_samples : int
        Subset size (capped at ``len(X)``).
    cmap_name : str
        Crameri colourmap for the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_seaborn()
    from sklearn.metrics import pairwise_distances

    setup_style()
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    dist_mat = pairwise_distances(X[idx], metric="cosine")
    cmap = get_crameri_cmap(cmap_name)

    g = sns.clustermap(
        dist_mat,
        cmap=cmap,
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
    )
    g.fig.suptitle(f"Cosine distance matrix ({len(idx)} random samples)", y=1.02)
    show_or_save(g.fig, save_path=save_path, show=show, close=True)
    return g.fig


# ===================================================================
#  Silhouette analysis
# ===================================================================


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Compute the mean silhouette score.

    Parameters
    ----------
    X : ndarray
    labels : ndarray
    metric : str

    Returns
    -------
    float
    """
    from sklearn.metrics import silhouette_score

    X = _ensure_2d(np.asarray(X))
    labels = np.asarray(labels)
    return float(silhouette_score(X, labels, metric=metric))


def plot_silhouette_per_group(
    X: np.ndarray,
    labels: np.ndarray,
    group_col_name: str = "Subtype",
    metric: str = "euclidean",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Box-plot of per-sample silhouette values grouped by label.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates (e.g. UMAP 2-D or raw 512-D).
    labels : ndarray
        Categorical group labels (same length as *X*).
    group_col_name : str
        Axis label / legend title.
    metric : str
        Distance metric for ``silhouette_samples``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_seaborn()
    from sklearn.metrics import silhouette_samples

    setup_style()
    X = _ensure_2d(np.asarray(X))
    labels = np.asarray(labels)
    sil_vals = silhouette_samples(X, labels, metric=metric)

    palette = _build_palette(labels, cmap_name)
    tmp = pd.DataFrame({group_col_name: labels, "silhouette": sil_vals})

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=tmp,
        x=group_col_name,
        y="silhouette",
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"Silhouette scores per {group_col_name}")
    ax.set_ylabel("Silhouette coefficient")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Hierarchical clustering dendrogram
# ===================================================================


def plot_dendrogram(
    X: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 150,
    metric: str = "cosine",
    method: str = "ward",
    color_threshold_ratio: float = 0.7,
    figsize: Tuple[float, float] = (10, 6),
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Ward-linkage dendrogram on a random subset.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    labels : ndarray
        Used as leaf labels.
    n_samples : int
        Subset size.
    metric : str
        Pair-wise distance metric.
    method : str
        Linkage method (``"ward"``, ``"average"``, …).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist

    setup_style()
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = _ensure_2d(X[idx])
    labels_sub = np.asarray(labels)[idx]

    dist_vec = pdist(X_sub, metric=metric)
    Z = hierarchy.linkage(dist_vec, method=method)

    fig, ax = plt.subplots(figsize=figsize)
    hierarchy.dendrogram(
        Z,
        labels=labels_sub,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold_ratio * float(Z[:, 2].max()),
        ax=ax,
    )
    ax.set_title(
        f"{method.title()} hierarchical clustering ({metric}) – "
        f"{len(idx)} random samples"
    )
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  K-Means confusion matrix
# ===================================================================


def plot_kmeans_confusion(
    X: np.ndarray,
    true_labels: np.ndarray,
    k: int = 5,
    random_state: int = 42,
    cmap_name: str = SEQUENTIAL_CMAP,
    figsize: Tuple[float, float] = (7, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Fit K-Means and show confusion matrix vs *true_labels*.

    Parameters
    ----------
    X : ndarray
        Feature matrix (raw or projected).
    true_labels : ndarray
        Ground-truth categorical labels.
    k : int
        Number of clusters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_seaborn()
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix

    setup_style()
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(_ensure_2d(X))

    unique_true = sorted(np.unique(true_labels))
    conf_mat = confusion_matrix(true_labels, cluster_labels, labels=unique_true)
    conf_df = pd.DataFrame(
        conf_mat,
        index=unique_true,
        columns=[f"Cluster_{i}" for i in range(k)],
    )

    cmap = get_crameri_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conf_df, annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_title("Confusion matrix: true labels vs K-means clusters")
    ax.set_ylabel("True label")
    ax.set_xlabel("Cluster")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig
