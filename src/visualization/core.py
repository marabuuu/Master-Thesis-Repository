"""
Visualization Core Utilities
=============================

Shared styling, Crameri scientific colour maps, and figure save/show helpers.
Used across all visualization sub-modules in the pipeline.

Colour maps
-----------
We use Fabio Crameri's perceptually uniform, colour-vision-deficiency-friendly
scientific colour maps (``cmcrameri`` package).  The module exposes a small
curated palette for the most common use-cases (categorical subtypes,
sequential heatmaps, diverging diff-maps) while allowing any ``cmc.*``
map to be requested by name.

Reference:  Crameri, F. (2018). Scientific colour maps. Zenodo.
            https://doi.org/10.5281/zenodo.1243862
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy / guarded imports so help(module) still works without heavy deps
# ---------------------------------------------------------------------------
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap, ListedColormap
    from matplotlib.figure import Figure

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

try:
    from cmcrameri import cm as cmc

    HAS_CRAMERI = True
except ImportError:  # pragma: no cover
    HAS_CRAMERI = False

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:  # pragma: no cover
    HAS_SNS = False


# ===================================================================
#  Crameri colour-map helpers
# ===================================================================

# Curated defaults – override via function arguments
CATEGORICAL_CMAP = "romaO"  # qualitative / categorical (alternating diverging for good contrast)
SEQUENTIAL_CMAP = "batlow"  # sequential heatmaps
DIVERGING_CMAP = "vik"  # diverging (e.g. fold-change)
HEATMAP_CMAP = "lajolla"  # single-hue heatmaps (e.g. distance matrices)


def get_crameri_cmap(name: str) -> Colormap:
    """Return a Crameri ``Colormap`` by name, falling back to matplotlib.

    Parameters
    ----------
    name : str
        Crameri map name (e.g. ``"batlow"``, ``"vikO"``) **or** any
        matplotlib colourmap name as a fallback.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    _check_matplotlib()
    if HAS_CRAMERI and hasattr(cmc, name):
        return getattr(cmc, name)
    # Fallback: try matplotlib's registry (includes 'cmc.*' names once
    # cmcrameri is imported)
    return mpl.colormaps[name]


def get_categorical_colors(
    n: int,
    cmap_name: str = CATEGORICAL_CMAP,
) -> List[str]:
    """Sample *n* distinct hex colours from a Crameri categorical map.

    Parameters
    ----------
    n : int
        Number of colours needed.
    cmap_name : str
        Name of a Crameri categorical (``*S``) colourmap, e.g. ``"batlowS"``.

    Returns
    -------
    list[str]
        Hex colour strings suitable for matplotlib / seaborn / plotly.
    """
    _check_matplotlib()
    cmap = get_crameri_cmap(cmap_name)
    indices = np.linspace(0, 1, n, endpoint=True)
    return [mpl.colors.to_hex(cmap(i)) for i in indices]


def build_label_palette(
    labels: np.ndarray,
    cmap_name: str = CATEGORICAL_CMAP,
) -> Dict[str, str]:
    """Map unique label strings to Crameri hex colours.
    
    For categorical data, uses strategic spacing to ensure maximum visual contrast
    between distinct categories.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of label values (strings or objects).
    cmap_name : str
        Name of Crameri categorical colourmap (default: CATEGORICAL_CMAP).
        
    Returns
    -------
    dict
        Dictionary mapping unique label strings to hex colour codes.
    """
    unique = sorted(np.unique(labels))
    n = len(unique)
    
    # For categorical data, use strategic spacing to maximize visual contrast
    # Avoid extremes (0, 1) and spread categories across colormap
    if n == 1:
        indices = [0.5]
    elif n == 2:
        indices = [0.15, 0.85]  # far apart
    elif n == 3:
        indices = [0.1, 0.5, 0.9]  # spread across
    elif n == 4:
        indices = [0.1, 0.35, 0.65, 0.9]  # evenly spaced
    elif n == 5:
        indices = [0.05, 0.3, 0.5, 0.7, 0.95]  # zig-zag for contrast
    elif n <= 10:
        # For larger category counts, use non-uniform spacing
        # Start with aggressive outer points, then fill interior
        indices = np.concatenate([
            np.linspace(0.05, 0.3, n//3, endpoint=False),
            np.linspace(0.35, 0.65, n//3, endpoint=False),
            np.linspace(0.7, 0.95, n - 2*(n//3))
        ])
    else:
        # For many categories, use linear spacing but avoid extremes
        indices = np.linspace(0.05, 0.95, n)
    
    cmap = get_crameri_cmap(cmap_name)
    colours = [mpl.colors.to_hex(cmap(float(i))) for i in indices]
    return dict(zip(unique, colours))


# ===================================================================
#  Plot styling
# ===================================================================

# Sensible defaults – may be overridden before calling any plot function.
_DEFAULT_RC: Dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def setup_style(
    style: str = "whitegrid",
    extra_rc: Optional[Dict[str, Any]] = None,
) -> None:
    """Apply a consistent plot style using seaborn + custom rcParams.

    Parameters
    ----------
    style : str
        Seaborn style name (``"whitegrid"``, ``"darkgrid"``, …).
    extra_rc : dict, optional
        Additional matplotlib rcParams to merge on top of defaults.
    """
    _check_matplotlib()
    if HAS_SNS:
        sns.set_theme(style=style, rc=_DEFAULT_RC)
    else:
        plt.rcParams.update(_DEFAULT_RC)
    if extra_rc:
        plt.rcParams.update(extra_rc)


# ===================================================================
#  Figure helpers
# ===================================================================


def save_figure(
    fig: Figure,
    save_path: Union[str, Path],
    close: bool = True,
    **savefig_kw: Any,
) -> Path:
    """Save a matplotlib figure and optionally close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : str | Path
        Destination path (parent dirs are created automatically).
    close : bool
        Close the figure after saving to free memory.
    **savefig_kw
        Forwarded to ``fig.savefig()``.

    Returns
    -------
    Path
        Resolved path that was written.
    """
    _check_matplotlib()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    defaults: Dict[str, Any] = dict(
        bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    defaults.update(savefig_kw)
    fig.savefig(save_path, **defaults)
    print(f"[OK] Saved figure → {save_path}")
    if close:
        plt.close(fig)
    return save_path


def show_or_save(
    fig: Figure,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    close: bool = True,
    **savefig_kw: Any,
) -> Optional[Figure]:
    """Convenience: save first (if requested), then show or return figure.

    Parameters
    ----------
    fig : Figure
    save_path : str | Path | None
    show : bool
    close : bool
    **savefig_kw
        Forwarded to :func:`save_figure`.

    Returns
    -------
    Figure | None
        Returns the figure only when *show* is ``False`` and *save_path*
        is ``None`` (i.e. the caller wants to do something custom).
    """
    if save_path is not None:
        save_figure(fig, save_path, close=False, **savefig_kw)
    if show:
        plt.show()
        if close:
            plt.close(fig)
        return None
    if save_path is not None and close:
        plt.close(fig)
        return None
    return fig


# ===================================================================
#  Guard helpers
# ===================================================================


def _check_matplotlib() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with:  pip install matplotlib"
        )


def _check_seaborn() -> None:
    _check_matplotlib()
    if not HAS_SNS:
        raise ImportError(
            "seaborn is required for this plot. "
            "Install it with:  pip install seaborn"
        )


def _check_crameri() -> None:
    if not HAS_CRAMERI:
        raise ImportError(
            "cmcrameri is required for Crameri colour maps. "
            "Install it with:  pip install cmcrameri"
        )
