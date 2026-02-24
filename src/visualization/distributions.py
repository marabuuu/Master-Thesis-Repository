"""
Distribution Plots
==================

Reusable bar-charts, count-plots, stacked-bar compositions, and mosaic
plots for categorical data.  These are useful throughout the pipeline
(e.g. subtype distributions, train/test splits, cluster compositions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .core import (
    CATEGORICAL_CMAP,
    _check_matplotlib,
    _check_seaborn,
    get_categorical_colors,
    get_crameri_cmap,
    setup_style,
    show_or_save,
)

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
#  Count plots
# ===================================================================


def plot_countplot(
    df: pd.DataFrame,
    x: str,
    title: str = "Sample counts",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (8, 4),
    order_by_count: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **countplot_kw: Any,
) -> Figure:
    """Seaborn count-plot with Crameri colours.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Column to count.
    title : str
    cmap_name : str
    order_by_count : bool
        If True, bars are sorted by descending count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_seaborn()
    setup_style()

    order = (
        df[x].value_counts().index.tolist() if order_by_count else None
    )
    n_cats = df[x].nunique()
    palette = get_categorical_colors(n_cats, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        data=df,
        x=x,
        order=order,
        palette=palette,
        ax=ax,
        **countplot_kw,
    )
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Stacked bar (group × split)
# ===================================================================


def plot_stacked_bar(
    df: pd.DataFrame,
    group_col: str,
    stack_col: str,
    title: str = "Composition",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Stacked bar chart showing *stack_col* composition per *group_col*.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        X-axis categories (e.g. subtype).
    stack_col : str
        Stacked segments (e.g. train / test).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()
    setup_style()

    counts = (
        df.groupby([group_col, stack_col])
        .size()
        .reset_index(name="cnt")
    )
    pivot = counts.pivot(index=group_col, columns=stack_col, values="cnt").fillna(0)

    n_stacks = pivot.shape[1]
    colors = get_categorical_colors(n_stacks, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", stacked=True, color=colors, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Mosaic (optional – degrades gracefully)
# ===================================================================


def plot_mosaic(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Mosaic plot",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[Figure]:
    """Mosaic plot (requires ``statsmodels``).

    Returns ``None`` silently if statsmodels is not installed.
    """
    _check_matplotlib()
    setup_style()

    try:
        from statsmodels.graphics.mosaicplot import mosaic
    except ImportError:
        print("Mosaic plot unavailable (install statsmodels).")
        return None

    fig, _ = plt.subplots()
    mosaic(df, columns, ax=fig.gca())
    fig.gca().set_title(title)
    fig.tight_layout()

    show_or_save(fig, save_path=save_path, show=show)
    return fig
