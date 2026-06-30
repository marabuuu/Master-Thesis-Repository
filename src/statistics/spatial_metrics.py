#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial metrics for cell-type point patterns in histology tiles.

Metrics implemented:
  • Ripley's K / L-function: clustering vs. inhibition at multiple scales
  • Pair correlation function: local-scale spatial association
  • Voronoi cell area: local dispersion of each cell type
  • kNN graph: degree, clustering coefficient, connected components
  • Spatial autocorrelation: Moran's I, Geary's C (per cell-type indicator maps)

Usage:
  from src.statistics.spatial_metrics import compute_ripley_l, compute_voronoi_areas, compute_knn_metrics
  
  # Point pattern: (N, 2) array of (x, y) centroids
  points = ...  # from segmentation mask
  
  # Ripley's L-function (deviation from complete spatial randomness)
  radii, L_vals, L_envelope = compute_ripley_l(points, radii=np.arange(10, 200, 10), n_boots=99)
  
  # Voronoi cell areas (per point)
  areas = compute_voronoi_areas(points, bounding_box=(512, 512))
  
  # kNN graph metrics (degree, clustering coeff, component sizes)
  metrics = compute_knn_metrics(points, k=5)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree, Voronoi
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


# ===================================================================
# § 1  Ripley's K / L-function
# ===================================================================

def _csr_expected_K(r: np.ndarray, intensity: float) -> np.ndarray:
    """Expected K-function under complete spatial randomness (Poisson).
    
    K(r) = π·r² for CSR.
    """
    return np.pi * r**2


def compute_ripley_K(
    points: np.ndarray,
    radii: Optional[np.ndarray] = None,
    edge_correction: str = "toroidal",
    window: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Ripley's K-function for a 2D point pattern.

    Parameters
    ----------
    points : (N, 2) array
        (x, y) coordinates of point locations.
    radii : (M,) array or None
        Radii at which to evaluate K(r). If None, uses np.arange(5, 150, 5).
    edge_correction : str
        "toroidal" (default): wrap boundaries; "none": ignore edge effects.
    window : (width, height) or None
        Observation window dimensions. When provided, area and toroidal wrapping
        use these fixed tile dimensions instead of the bounding box of the
        observed points. Always pass this when the points come from a tile of
        known size so that K(r) is normalised correctly and the toroidal domain
        is consistent across all tiles and CSR simulations.

    Returns
    -------
    radii : (M,) array
        Radii queried.
    K_vals : (M,) array
        Ripley's K estimates.
    """
    if radii is None:
        radii = np.arange(5, 150, 5).astype(float)
    else:
        radii = np.asarray(radii, dtype=float)

    points = np.asarray(points, dtype=float)
    N = points.shape[0]

    if N < 2:
        return radii, np.zeros_like(radii)

    if window is not None:
        xmin, ymin = 0.0, 0.0
        width, height = float(window[0]), float(window[1])
    else:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        width  = xmax - xmin
        height = ymax - ymin
    area = width * height

    # Precompute all pairwise distances once (upper triangle only).
    if edge_correction == "toroidal":
        dx = np.abs(points[:, None, 0] - points[None, :, 0])
        dy = np.abs(points[:, None, 1] - points[None, :, 1])
        dx = np.minimum(dx, width - dx)
        dy = np.minimum(dy, height - dy)
        triu = np.triu_indices(N, k=1)
        pair_dists = np.sqrt(dx**2 + dy**2)[triu]
    else:
        pair_dists = pdist(points)

    K_vals = area / (N**2) * 2 * np.array([np.sum(pair_dists <= r) for r in radii])

    return radii, K_vals


def compute_ripley_L(
    points: np.ndarray,
    radii: Optional[np.ndarray] = None,
    edge_correction: str = "toroidal",
    window: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Ripley's L-function: L(r) = sqrt(K(r) / π) - r.

    L(r) = 0 under CSR, < 0 indicates inhibition, > 0 indicates clustering.

    Parameters
    ----------
    points : (N, 2) array
        Point coordinates.
    radii : (M,) array or None
        Radii to evaluate. If None, auto-computed.
    edge_correction : str
        "toroidal" or "none".
    window : (width, height) or None
        Observation window; passed through to compute_ripley_K.

    Returns
    -------
    radii : (M,) array
    L_vals : (M,) array
        L(r) values (negative = inhibition, positive = clustering).
    """
    radii, K_vals = compute_ripley_K(points, radii, edge_correction, window=window)
    L_vals = np.sqrt(K_vals / np.pi) - radii
    return radii, L_vals


def compute_ripley_L_bootstrap(
    points: np.ndarray,
    radii: Optional[np.ndarray] = None,
    n_bootstrap: int = 99,
    edge_correction: str = "toroidal",
    seed: int = 42,
    window: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Ripley's L-function with bootstrap confidence intervals under CSR.

    Simulates n_bootstrap point patterns with the same intensity (N / area) as the
    observed data, uniformly distributed in the observation window, and computes
    L(r) for each simulation. Returns percentile-based envelope.

    Parameters
    ----------
    points : (N, 2) array
        Observed point pattern.
    radii : (M,) array or None
        Radii to evaluate.
    n_bootstrap : int
        Number of CSR simulations (default 99 → 2.5% and 97.5% quantiles).
    edge_correction : str
    seed : int
    window : (width, height) or None
        Observation window dimensions. When provided, both the observed L(r) and
        all CSR simulations use this fixed domain, ensuring a consistent
        comparison. Without it the observed pattern uses its own point bbox while
        each CSR simulation uses a slightly smaller bbox (uniform samples never
        exactly reach the extremes), introducing a systematic downward bias in
        the envelope.

    Returns
    -------
    radii : (M,) array
    L_obs : (M,) array
        Observed L values.
    L_lower : (M,) array
        Lower 2.5% quantile of CSR simulations.
    L_upper : (M,) array
        Upper 97.5% quantile of CSR simulations.
    """
    rng = np.random.default_rng(seed)
    radii, L_obs = compute_ripley_L(points, radii, edge_correction, window=window)

    points = np.asarray(points)
    N = points.shape[0]

    if window is not None:
        sim_xmin, sim_ymin = 0.0, 0.0
        sim_xmax, sim_ymax = float(window[0]), float(window[1])
    else:
        sim_xmin, sim_ymin = points.min(axis=0)
        sim_xmax, sim_ymax = points.max(axis=0)

    L_boots = []
    for _ in range(n_bootstrap):
        x_sim = rng.uniform(sim_xmin, sim_xmax, N)
        y_sim = rng.uniform(sim_ymin, sim_ymax, N)
        points_sim = np.column_stack([x_sim, y_sim])
        _, L_sim = compute_ripley_L(points_sim, radii, edge_correction, window=window)
        L_boots.append(L_sim)

    L_boots = np.array(L_boots)
    L_lower = np.percentile(L_boots, 2.5, axis=0)
    L_upper = np.percentile(L_boots, 97.5, axis=0)

    return radii, L_obs, L_lower, L_upper


# ===================================================================
# § 2  Voronoi cell areas
# ===================================================================

def compute_voronoi_areas(
    points: np.ndarray,
    bounding_box: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Compute Voronoi cell area for each point.
    
    For points near the boundary, areas are clipped to the bounding box.
    Points outside the bounding box or with infinite Voronoi regions
    are assigned NaN.
    
    Parameters
    ----------
    points : (N, 2) array
        (x, y) coordinates.
    bounding_box : (width, height) or None
        Bounding box for clipping Voronoi regions.
        If None, uses min/max of points with 10% padding.
    
    Returns
    -------
    areas : (N,) array
        Voronoi cell area for each point. NaN for unbounded cells.
    """
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    
    if N < 3:
        return np.full(N, np.nan)
    
    if bounding_box is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        width = xmax - xmin
        height = ymax - ymin
        padding = 0.1
        xmin -= width * padding
        xmax += width * padding
        ymin -= height * padding
        ymax += height * padding
    else:
        xmin, ymin = 0, 0
        xmax, ymax = bounding_box
    
    try:
        vor = Voronoi(points)
    except Exception as e:
        logger.warning("Voronoi computation failed: %s", e)
        return np.full(N, np.nan)
    
    areas = np.full(N, np.nan)
    
    for point_idx in range(N):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        
        # Skip if unbounded (contains -1)
        if -1 in region:
            continue
        
        if len(region) < 3:
            continue
        
        # Get Voronoi vertices and clip to bounding box
        vertices = vor.vertices[region]
        vertices = np.clip(vertices, [xmin, ymin], [xmax, ymax])
        
        # Compute area using shoelace formula
        if len(vertices) >= 3:
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.sum(x * np.roll(y, 1) - np.roll(x, 1) * y))
            areas[point_idx] = area
    
    return areas


# ===================================================================
# § 3  kNN graph metrics
# ===================================================================

def compute_knn_metrics(
    points: np.ndarray,
    k: int = 5,
) -> Dict[str, np.ndarray | float]:
    """Compute k-nearest-neighbour graph metrics.
    
    Parameters
    ----------
    points : (N, 2) array
        Point coordinates.
    k : int
        Number of nearest neighbours (default 5).
    
    Returns
    -------
    metrics : dict
        'degree' : (N,) array — degree per node (always k for non-boundary)
        'mean_degree' : float
        'clustering_coeff' : (N,) array — local clustering coefficient
        'mean_clustering_coeff' : float
        'component_sizes' : list[int] — connected component sizes
        'n_components' : int
        'largest_component_frac' : float — fraction of nodes in largest component
        'mean_knn_dist' : (N,) array — mean distance to k-NN
        'global_mean_knn_dist' : float
    """
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    
    if N < k + 1:
        return {
            'degree': np.full(N, np.nan),
            'mean_degree': np.nan,
            'clustering_coeff': np.full(N, np.nan),
            'mean_clustering_coeff': np.nan,
            'component_sizes': [N],
            'n_components': 1,
            'largest_component_frac': 1.0,
            'mean_knn_dist': np.full(N, np.nan),
            'global_mean_knn_dist': np.nan,
        }
    
    # Build kNN graph using KDTree
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)  # +1 to exclude self
    
    # Remove self (first neighbor is always the point itself)
    knn_indices = indices[:, 1:]
    knn_distances = distances[:, 1:]
    
    # Degree and mean kNN distance
    degree = np.full(N, k, dtype=int)
    mean_knn_dist = np.mean(knn_distances, axis=1)
    
    # Build adjacency matrix (directed, k-NN)
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in knn_indices[i]:
            adj[i, j] = 1
    
    # Local clustering coefficient: C_i = |edges between neighbors| / max_possible
    clustering_coeff = np.zeros(N)
    for i in range(N):
        neighbors = knn_indices[i]
        n_neighbors = len(neighbors)
        if n_neighbors < 2:
            clustering_coeff[i] = 0
        else:
            # Count edges between neighbors
            edges_between = 0
            for j in range(n_neighbors):
                for m in range(j + 1, n_neighbors):
                    # Undirected: edge if either direction exists
                    if adj[neighbors[j], neighbors[m]] or adj[neighbors[m], neighbors[j]]:
                        edges_between += 1
            max_edges = n_neighbors * (n_neighbors - 1) / 2
            clustering_coeff[i] = edges_between / max_edges if max_edges > 0 else 0
    
    # Connected components (treat as undirected for components)
    adj_undirected = np.logical_or(adj, adj.T).astype(int)
    visited = np.zeros(N, dtype=bool)
    components = []
    
    for start in range(N):
        if visited[start]:
            continue
        # BFS to find component
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in np.where(adj_undirected[node])[0]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        components.append(len(component))
    
    component_sizes = sorted(components, reverse=True)
    largest_frac = component_sizes[0] / N if component_sizes else 0
    
    return {
        'degree': degree,
        'mean_degree': float(np.mean(degree)),
        'clustering_coeff': clustering_coeff,
        'mean_clustering_coeff': float(np.mean(clustering_coeff)),
        'component_sizes': component_sizes,
        'n_components': len(component_sizes),
        'largest_component_frac': largest_frac,
        'mean_knn_dist': mean_knn_dist,
        'global_mean_knn_dist': float(np.mean(mean_knn_dist)),
    }


# ===================================================================
# § 4  Pair correlation function g(r)
# ===================================================================

def compute_pair_correlation(
    points: np.ndarray,
    radii: Optional[np.ndarray] = None,
    n_bins: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical pair correlation function g(r).
    
    g(r) = (mean number of points at distance r) / (expected under CSR).
    g(r) ≈ 1 under CSR, < 1 for inhibition, > 1 for clustering.
    
    Parameters
    ----------
    points : (N, 2) array
    radii : (M,) array or None
        Radii to evaluate. If None, auto-computed from inter-point distances.
    n_bins : int
        If radii is None, divide the distance range into this many bins.
    
    Returns
    -------
    r_centers : (M,) array
    g_r : (M,) array
    """
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    
    if N < 2:
        return np.array([]), np.array([])
    
    # Compute all pairwise distances
    dists = pdist(points)
    
    if radii is None:
        r_max = np.max(dists)
        radii = np.linspace(0, r_max, n_bins + 1)
    else:
        radii = np.asarray(radii, dtype=float)
    
    r_centers = (radii[:-1] + radii[1:]) / 2
    g_r = np.zeros(len(r_centers))
    
    # Bounding box
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    area = (xmax - xmin) * (ymax - ymin)
    intensity = N / area
    
    for idx, (r_lo, r_hi) in enumerate(zip(radii[:-1], radii[1:])):
        # Count pairs in annulus [r_lo, r_hi)
        count = np.sum((dists >= r_lo) & (dists < r_hi))
        
        # Expected count under CSR: intensity² × annulus area × 2
        # (factor of 2 accounts for undirected pairs)
        dr = r_hi - r_lo
        annulus_area = np.pi * (r_hi**2 - r_lo**2)
        expected = 2 * intensity**2 * annulus_area * (N * (N - 1) / 2)
        
        g_r[idx] = count / expected if expected > 0 else np.nan
    
    return r_centers, g_r


# ===================================================================
# § 5  Spatial autocorrelation (Moran's I)
# ===================================================================

def compute_morans_I(
    points: np.ndarray,
    values: np.ndarray,
    distance_threshold: float = 50.0,
) -> float:
    """Compute Moran's I spatial autocorrelation index.
    
    I > 0: values of nearby points are similar (clustering).
    I < 0: values of nearby points are dissimilar (dispersion).
    I ≈ 0: random spatial association.
    
    Parameters
    ----------
    points : (N, 2) array
        Point coordinates.
    values : (N,) array
        Values at each point (e.g., 0/1 indicator for cell type presence).
    distance_threshold : float
        Spatial weight threshold; pairs within this distance are considered neighbors.
    
    Returns
    -------
    morans_I : float
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    N = len(values)
    
    if N < 2:
        return np.nan
    
    # Compute pairwise distances
    dists = pdist(points)
    dists_sq = squareform(dists)
    
    # Weight matrix: 1 if distance <= threshold, 0 otherwise
    W = (dists_sq <= distance_threshold).astype(float)
    np.fill_diagonal(W, 0)
    
    W_sum = np.sum(W)
    if W_sum == 0:
        return np.nan
    
    # Centre values around mean
    values_c = values - np.mean(values)
    
    # Moran's I = (N / W_sum) × (sum_ij W_ij × y_i × y_j) / (sum_i y_i²)
    numerator = np.sum(W * np.outer(values_c, values_c))
    denominator = np.sum(values_c**2)
    
    morans_I = (N / W_sum) * (numerator / denominator) if denominator > 0 else np.nan
    
    return float(morans_I)
