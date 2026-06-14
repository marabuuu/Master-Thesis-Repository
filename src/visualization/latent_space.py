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
    build_label_palette,
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


def plot_pca_variance(
    X: np.ndarray,
    n_components: int = 20,
    cmap_name: str = SEQUENTIAL_CMAP,
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    pca_kw: Optional[Dict[str, Any]] = None,
) -> Tuple[Figure, Any]:
    """PCA scree plot: per-component and cumulative explained variance.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    n_components : int
        Number of components to inspect (capped at min(n, d)).
    cmap_name : str
        Crameri sequential colourmap for the bar chart.
    figsize : tuple
    save_path, show : standard output options.
    pca_kw : dict, optional
        Extra keyword arguments forwarded to ``sklearn.decomposition.PCA``.

    Returns
    -------
    (fig, pca)  – figure and the fitted ``sklearn`` PCA object.
    """
    _check_matplotlib()
    from sklearn.decomposition import PCA

    setup_style()
    X = _ensure_2d(X)
    n = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n, **(pca_kw or {}))
    pca.fit(X)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    cmap = get_crameri_cmap(cmap_name)
    bar_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: per-component bar chart
    axes[0].bar(range(1, n + 1), var_ratio * 100, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Variance")
    tick_step = max(1, n // 10)
    axes[0].set_xticks(range(1, n + 1, tick_step))
    axes[0].grid(True, alpha=0.25, axis="y")

    # Right: cumulative variance line
    c_line = cmap(0.75)
    axes[1].plot(range(1, n + 1), cum_var * 100, "-o", markersize=4,
                 linewidth=1.5, color=c_line)
    axes[1].axhline(90, color="firebrick", linestyle="--", alpha=0.6, label="90%")
    axes[1].axhline(95, color="darkorange", linestyle=":", alpha=0.6, label="95%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Variance")
    axes[1].set_ylim(0, 105)
    axes[1].legend(framealpha=0.9)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("PCA Explained Variance", fontweight="bold")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig, pca


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
    palette: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    hue_title: str = "hue",
    style_title: str = "style",
    label_rename: Optional[Dict[str, str]] = None,
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

    if label_rename:
        hue_labels = np.array([label_rename.get(str(l), str(l)) for l in hue_labels])
        if style_labels is not None:
            style_labels = np.array([label_rename.get(str(l), str(l)) for l in style_labels])
        if palette is not None:
            palette = {label_rename.get(k, k): v for k, v in palette.items()}
        if markers is not None:
            markers = {label_rename.get(k, k): v for k, v in markers.items()}

    if palette is None:
        palette = build_label_palette(hue_labels, cmap_name)
    else:
        # Treat palette as override: auto-fill any labels not covered by cmap
        missing = set(np.unique(hue_labels)) - set(palette.keys())
        if missing:
            auto = build_label_palette(hue_labels, cmap_name)
            palette = {**auto, **palette}

    tmp = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], hue_title: hue_labels}
    )
    if style_labels is not None:
        tmp[style_title] = style_labels

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=tmp,
        x="x",
        y="y",
        hue=hue_title,
        style=style_title if style_labels is not None else None,
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
    if title:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig


def plot_umap(
    X: np.ndarray,
    hue_labels: np.ndarray,
    style_labels: Optional[np.ndarray] = None,
    title: str = "",
    cmap_name: str = CATEGORICAL_CMAP,
    palette: Optional[Dict[str, str]] = None,
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
        palette=palette,
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
    title: str = "",
    cmap_name: str = CATEGORICAL_CMAP,
    palette: Optional[Dict[str, str]] = None,
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
        palette=palette,
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
    palette: Optional[Dict[str, str]] = None,
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
    if title is None:
        title = ""
    fig = plot_projection(
        X_pca,
        hue_labels,
        style_labels=style_labels,
        title=title,
        axis_labels=("PC1", "PC2"),
        cmap_name=cmap_name,
        palette=palette,
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
    labels: Optional[np.ndarray] = None,
    n_samples: int = 200,
    cmap_name: str = HEATMAP_CMAP,
    cat_cmap_name: str = CATEGORICAL_CMAP,
    palette: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 10),
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Hierarchically-clustered cosine-distance heatmap on a random subset.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    labels : ndarray, optional
        Categorical labels to plot as color bars along the axes.
    n_samples : int
        Subset size (capped at ``len(X)``).
    cmap_name : str
        Crameri colourmap for the heatmap.
    cat_cmap_name : str
        Crameri colormap for the label color bar.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_seaborn()
    from sklearn.metrics import pairwise_distances

    setup_style()
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    
    sub_X = X[idx]
    dist_mat = pairwise_distances(sub_X, metric="cosine")
    cmap = get_crameri_cmap(cmap_name)
    
    # Handle label color bars if labels are provided
    row_colors = None
    _pal = palette
    if labels is not None:
        sub_labels = labels[idx]
        if _pal is None:
            _pal = build_label_palette(sub_labels, cat_cmap_name)
        else:
            missing = set(np.unique(sub_labels)) - set(_pal.keys())
            if missing:
                auto = build_label_palette(sub_labels, cat_cmap_name)
                _pal = {**auto, **_pal}
        row_colors = pd.Series(sub_labels).map(_pal).values

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.clustermap(
            dist_mat,
            cmap=cmap,
            figsize=figsize,
            row_colors=row_colors,
            col_colors=row_colors,
            xticklabels=False,
            yticklabels=False,
            dendrogram_ratio=0.01,
            cbar_pos=(1.02, 0.1, 0.03, 0.5),
        )

    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

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
    palette: Optional[Dict[str, str]] = None,
    label_rename: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (8, 5),
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

    if palette is None:
        _pal = build_label_palette(labels, cmap_name)
    else:
        missing = set(np.unique(labels)) - set(palette.keys())
        _pal = {**build_label_palette(labels, cmap_name), **palette} if missing else palette
    tmp = pd.DataFrame({group_col_name: labels, "silhouette": sil_vals})

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=tmp,
        x=group_col_name,
        y="silhouette",
        palette=_pal,
        ax=ax,
    )
    if label_rename:
        ax.set_xticklabels([label_rename.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.set_ylabel("Silhouette coefficient", fontsize=13)
    ax.set_xlabel("", fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_ylim(-1, 1)
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
    palette: Optional[Dict[str, str]] = None,
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

    # Build per-node cohort map for palette-driven branch coloring
    if palette is not None:
        n_leaves = len(labels_sub)
        node_cohort: Dict[int, str] = {i: str(labels_sub[i]) for i in range(n_leaves)}
        for i, row in enumerate(Z):
            left, right = int(row[0]), int(row[1])
            lc, rc = node_cohort.get(left), node_cohort.get(right)
            node_cohort[n_leaves + i] = lc if lc == rc else "mixed"

        def _link_color_func(k: int) -> str:
            return palette.get(node_cohort.get(k, "mixed"), "#888888")

    fig, ax = plt.subplots(figsize=figsize)
    d = hierarchy.dendrogram(
        Z,
        labels=labels_sub,
        leaf_rotation=90,
        leaf_font_size=0 if palette is not None else 8,
        color_threshold=0 if palette is not None else color_threshold_ratio * float(Z[:, 2].max()),
        link_color_func=_link_color_func if palette is not None else None,
        ax=ax,
    )

    # Color the x-tick labels by cohort when palette is provided
    if palette is not None:
        leaves = d["leaves"]
        for tick, leaf_idx in zip(ax.get_xticklabels(), leaves):
            tick.set_color(palette.get(str(labels_sub[leaf_idx]), "black"))
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=palette[l], label=l)
                   for l in sorted(set(str(lb) for lb in labels_sub)) if l in palette]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=11)

    ax.set_ylabel("Linkage distance", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
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


# ===================================================================
#  Composite Report Figure: Clustermap + UMAP + Silhouette
# ===================================================================


def plot_composite_latent_analysis(
    X: np.ndarray,
    coords_umap: np.ndarray,
    hue_labels: np.ndarray,
    style_labels: np.ndarray,
    cmap_name: str = CATEGORICAL_CMAP,
    heatmap_cmap_name: str = HEATMAP_CMAP,
    figsize: Tuple[float, float] = (18, 10),
    n_samples_heatmap: int = 200,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Create a composite report figure with three panels:
    
    - **Left (large)**: Cosine distance clustermap of raw features
    - **Top-right**: UMAP projection colored by subtype, marked by split
    - **Bottom-right**: Silhouette scores per subtype
    
    Single shared legend showing subtype colors + split marker shapes for report readability.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Raw feature matrix (e.g., 512-D VAE latents).
    coords_umap : ndarray, shape (n, 2)
        Pre-computed UMAP 2-D coordinates.
    hue_labels : ndarray
        Categorical labels for coloring (e.g., PAM50 subtypes).
    style_labels : ndarray
        Categorical labels for marker styles (e.g., train/val/test split).
    cmap_name : str
        Crameri categorical colormap for subtypes.
    heatmap_cmap_name : str
        Crameri colormap for the cosine distance heatmap.
    figsize : tuple
        Overall figure size (width, height).
    n_samples_heatmap : int
        Number of random samples to show in clustermap.
    random_state : int
        For reproducibility.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()
    _check_seaborn()
    from scipy.cluster import hierarchy
    from sklearn.metrics import silhouette_samples, pairwise_distances

    setup_style()

    # Build color palette for subtypes
    palette = build_label_palette(hue_labels, cmap_name)
    marker_map = {"train": "o", "test": "^", "val": "s", "unknown": "D"}

    # Main layout: left clustermap-style panel, middle two smaller plots, right legend column.
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[2.65, 1.2, 0.85],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.26,
    )

    # ──────────────────────────────────────────────────
    # LEFT PANEL: clustermap-like cosine distance view
    # ──────────────────────────────────────────────────
    left = outer[:, 0].subgridspec(
        3,
        3,
        width_ratios=[0.18, 0.035, 1.0],
        height_ratios=[0.18, 0.035, 1.0],
        wspace=0.01,
        hspace=0.01,
    )

    ax_row_dend = fig.add_subplot(left[2, 0])
    ax_col_dend = fig.add_subplot(left[0, 2])
    ax_row_colors = fig.add_subplot(left[2, 1])
    ax_col_colors = fig.add_subplot(left[1, 2])
    ax_heatmap = fig.add_subplot(left[2, 2])

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples_heatmap, len(X)), replace=False)
    X_sub = _ensure_2d(X[idx])
    labels_sub = np.asarray(hue_labels)[idx]
    dist_mat = pairwise_distances(X_sub, metric="cosine")

    # Use the exact same clustering path as the standalone clustermap plot.
    # This avoids subtle ordering differences compared with manual linkage choices.
    palette_sub = build_label_palette(labels_sub, cmap_name)
    row_colors_sub = pd.Series(labels_sub).map(palette_sub).values
    g_tmp = sns.clustermap(
        dist_mat,
        cmap=get_crameri_cmap(heatmap_cmap_name),
        row_colors=row_colors_sub,
        col_colors=row_colors_sub,
        xticklabels=False,
        yticklabels=False,
    )
    row_order = g_tmp.dendrogram_row.reordered_ind
    col_order = g_tmp.dendrogram_col.reordered_ind
    row_linkage = g_tmp.dendrogram_row.linkage
    col_linkage = g_tmp.dendrogram_col.linkage
    plt.close(g_tmp.fig)

    dist_reordered = dist_mat[np.ix_(row_order, col_order)]
    labels_ordered_row = labels_sub[row_order]
    labels_ordered_col = labels_sub[col_order]

    # Draw dendrograms (top and left).
    hierarchy.dendrogram(
        col_linkage,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#444444",
        ax=ax_col_dend,
    )
    ax_col_dend.set_xticks([])
    ax_col_dend.set_yticks([])
    for spine in ax_col_dend.spines.values():
        spine.set_visible(False)

    hierarchy.dendrogram(
        row_linkage,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#444444",
        ax=ax_row_dend,
    )
    ax_row_dend.set_xticks([])
    ax_row_dend.set_yticks([])
    for spine in ax_row_dend.spines.values():
        spine.set_visible(False)

    # Draw subtype color bars akin to row_colors/col_colors in standalone clustermap.
    color_rgb_row = np.array([
        plt.matplotlib.colors.to_rgb(palette[label]) for label in labels_ordered_row
    ])
    color_rgb_col = np.array([
        plt.matplotlib.colors.to_rgb(palette[label]) for label in labels_ordered_col
    ])

    ax_row_colors.imshow(color_rgb_row.reshape(-1, 1, 3), aspect="auto", interpolation="nearest")
    ax_row_colors.set_xticks([])
    ax_row_colors.set_yticks([])
    for spine in ax_row_colors.spines.values():
        spine.set_visible(False)

    ax_col_colors.imshow(color_rgb_col.reshape(1, -1, 3), aspect="auto", interpolation="nearest")
    ax_col_colors.set_xticks([])
    ax_col_colors.set_yticks([])
    for spine in ax_col_colors.spines.values():
        spine.set_visible(False)

    cmap_heat = get_crameri_cmap(heatmap_cmap_name)
    im = ax_heatmap.imshow(dist_reordered, cmap=cmap_heat, aspect="auto", interpolation="nearest")
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xlabel("Samples (cluster-ordered)", fontsize=11, fontweight="bold")
    ax_heatmap.set_ylabel("")

    ax_col_dend.set_title(
        f"Cosine Distance Clustermap ({len(idx)} random samples)",
        fontsize=12,
        fontweight="bold",
        pad=2,
    )

    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.035, pad=0.015)
    cbar.set_label("Cosine Distance", fontsize=10)

    # ──────────────────────────────────────────────────
    # MIDDLE TOP: UMAP projection
    # ──────────────────────────────────────────────────
    ax_umap = fig.add_subplot(outer[0, 1])

    # Scatter plot: one series per (subtype, split) combination for unified legend
    for split in sorted(np.unique(style_labels)):
        for subtype in sorted(np.unique(hue_labels)):
            mask = (hue_labels == subtype) & (style_labels == split)
            if mask.sum() > 0:
                ax_umap.scatter(
                    coords_umap[mask, 0],
                    coords_umap[mask, 1],
                    c=[palette[subtype]],
                    marker=marker_map.get(split, "o"),
                    s=60,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.5,
                    label=f"{subtype} ({split})",
                )

    ax_umap.set_xlabel("UMAP 1", fontsize=11, fontweight="bold")
    ax_umap.set_ylabel("UMAP 2", fontsize=11, fontweight="bold")
    ax_umap.set_title(
        "UMAP of Genomic Representation",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_umap.grid(True, alpha=0.2, linestyle="--")

    # ──────────────────────────────────────────────────
    # MIDDLE BOTTOM: Silhouette scores
    # ──────────────────────────────────────────────────
    ax_sil = fig.add_subplot(outer[1, 1])

    sil_vals = silhouette_samples(coords_umap, hue_labels, metric="euclidean")
    sil_df = pd.DataFrame({"Subtype": hue_labels, "Silhouette": sil_vals})

    sns.boxplot(
        data=sil_df,
        x="Subtype",
        y="Silhouette",
        hue="Subtype",
        palette=palette,
        dodge=False,
        ax=ax_sil,
        width=0.6,
    )
    if ax_sil.get_legend() is not None:
        ax_sil.get_legend().remove()

    ax_sil.set_xlabel("PAM50 Subtype", fontsize=11, fontweight="bold")
    ax_sil.set_ylabel("Silhouette Coefficient", fontsize=10, fontweight="bold", labelpad=6)
    ax_sil.yaxis.set_label_position("right")
    ax_sil.set_title(
        "Silhouette Scores per PAM50 Subtype",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_sil.set_ylim(-1.0, 1.0)
    ax_sil.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_sil.grid(True, alpha=0.2, axis="y", linestyle="--")

    # ──────────────────────────────────────────────────
    # RIGHT COLUMN: unified legend (no overlap)
    # ──────────────────────────────────────────────────
    ax_leg = fig.add_subplot(outer[:, 2])
    ax_leg.axis("off")

    legend_elements = []

    legend_elements.append(
        plt.Line2D([0], [0], color="none", label="Subtype Colors")
    )

    # Add subtype colors with marker='o'
    for subtype in sorted(np.unique(hue_labels)):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette[subtype],
                markersize=8,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=subtype,
            )
        )

    # Add spacer and second section label
    legend_elements.append(plt.Line2D([0], [0], color="none", label=""))
    legend_elements.append(
        plt.Line2D([0], [0], color="none", label="Data Split Markers")
    )

    # Add split marker styles
    for split in sorted(np.unique(style_labels)):
        marker = marker_map.get(split, "o")
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=8,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=split,
            )
        )

    ax_leg.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.0),
        ncol=1,
        frameon=True,
        fontsize=9,
        framealpha=0.95,
        edgecolor="k",
        handlelength=1.4,
        labelspacing=0.6,
    )

    fig.suptitle(
        "Latent Space Analysis: Genomic Representation and Subtype Separability",
        fontsize=14,
        fontweight="bold",
        y=0.988,
    )

    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.06)

    show_or_save(fig, save_path=save_path, show=show)
    return fig
