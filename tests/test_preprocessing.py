# -*- coding: utf-8 -*-
"""Tests for the preprocessing submodule.

Focus areas
-----------
* Data structures returned by preprocessing utilities (DataFrame vs ndarray, shape)
* GeneExpressionDataLoader dispatch of preprocessing modes
* Utility helpers: polygon creation, coordinate parsing, annotation reading,
  ZIP/CSV matching, MPP calculation
"""

import os
import tempfile
import zipfile
import json

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.utils import (
    preprocess_log1p_minmax,
    preprocess_log1p_zscore,
    inspect_variance,
    create_polygons,
    create_dataframe,
    extract_coordinates,
    parse_tile_coords,
    create_polygons_from_filenames,
    contains_nan_or_inf,
    fix_invalid_polygon,
    read_tiler_params,
    get_mpp_from_tiler_params,
    match_zip_to_csv,
    read_zip_tile_list,
    read_annotations,
)
from src.preprocessing.data_loader import GeneExpressionDataLoader


# ======================================================================
#   preprocess_log1p_zscore / preprocess_log1p_minmax
# ======================================================================

class TestPreprocessFunctions:
    """Verify that preprocessing helpers return correct types and shapes."""

    def test_log1p_zscore_returns_dataframe(self, raw_gene_expression_df):
        result = preprocess_log1p_zscore(raw_gene_expression_df)
        assert isinstance(result, pd.DataFrame), "Expected a DataFrame back"
        assert result.shape == raw_gene_expression_df.shape

    def test_log1p_zscore_produces_zero_mean_unit_var(self, raw_gene_expression_df):
        result = preprocess_log1p_zscore(raw_gene_expression_df)
        col_means = result.mean(axis=0)
        col_stds = result.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(col_means.values, 0.0, decimal=5)
        # Columns with non-zero variance should have std ≈ 1
        mask = col_stds > 0
        np.testing.assert_array_almost_equal(col_stds[mask].values, 1.0, decimal=5)

    def test_log1p_minmax_returns_dataframe(self, raw_gene_expression_df):
        result = preprocess_log1p_minmax(raw_gene_expression_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == raw_gene_expression_df.shape

    def test_log1p_minmax_values_in_unit_range(self, raw_gene_expression_df):
        result = preprocess_log1p_minmax(raw_gene_expression_df)
        assert result.values.min() >= 0.0 - 1e-9
        assert result.values.max() <= 1.0 + 1e-9

    def test_preserves_index_and_columns(self, raw_gene_expression_df):
        for fn in (preprocess_log1p_zscore, preprocess_log1p_minmax):
            result = fn(raw_gene_expression_df)
            pd.testing.assert_index_equal(result.index, raw_gene_expression_df.index)
            pd.testing.assert_index_equal(result.columns, raw_gene_expression_df.columns)


# ======================================================================
#   inspect_variance
# ======================================================================

class TestInspectVariance:

    def test_returns_dict_with_expected_keys(self, raw_gene_expression_df):
        result = inspect_variance(raw_gene_expression_df)
        assert isinstance(result, dict)
        for key in ("n_genes", "n_samples", "gene_var_summary",
                     "n_zero_var_genes", "sample_var_summary"):
            assert key in result, f"Missing key: {key}"

    def test_dimensions_match(self, raw_gene_expression_df):
        result = inspect_variance(raw_gene_expression_df)
        assert result["n_samples"] == raw_gene_expression_df.shape[0]
        assert result["n_genes"] == raw_gene_expression_df.shape[1]

    def test_zero_var_genes_count(self):
        """If a column is constant, it should be counted."""
        df = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 2.0, 3.0]})
        result = inspect_variance(df)
        assert result["n_zero_var_genes"] == 1


# ======================================================================
#   GeneExpressionDataLoader
# ======================================================================

class TestGeneExpressionDataLoader:

    def test_load_data_returns_dataframe(self, gene_expression_csv):
        loader = GeneExpressionDataLoader(gene_expression_csv)
        data = loader.load_data()
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] > 0 and data.shape[1] > 0

    def test_preprocess_none_returns_ndarray(self, gene_expression_csv):
        loader = GeneExpressionDataLoader(gene_expression_csv, preprocess_mode="none")
        data = loader.load_data()
        result = loader.preprocess_data(data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_preprocess_log_zscore_returns_ndarray(self, gene_expression_csv):
        loader = GeneExpressionDataLoader(gene_expression_csv, preprocess_mode="log_zscore")
        data = loader.load_data()
        result = loader.preprocess_data(data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_preprocess_log_minmax_bounded(self, gene_expression_csv):
        loader = GeneExpressionDataLoader(gene_expression_csv, preprocess_mode="log_minmax")
        data = loader.load_data()
        result = loader.preprocess_data(data)
        assert isinstance(result, np.ndarray)
        assert result.min() >= -1e-9
        assert result.max() <= 1.0 + 1e-9

    def test_preprocess_auto_with_raw_counts(self, gene_expression_csv):
        """Auto mode on raw counts should apply log1p + zscore."""
        loader = GeneExpressionDataLoader(gene_expression_csv, preprocess_mode="auto")
        data = loader.load_data()
        result = loader.preprocess_data(data)
        assert isinstance(result, np.ndarray)
        # After z-scoring, column means should be near 0
        col_means = result.mean(axis=0)
        assert np.abs(col_means).mean() < 0.5  # rough sanity check

    def test_columns_to_drop(self, gene_expression_csv_with_label):
        loader = GeneExpressionDataLoader(
            gene_expression_csv_with_label,
            columns_to_drop=["Majority_Subtype_mRNA"],
            preprocess_mode="none",
        )
        data = loader.load_data()
        result = loader.preprocess_data(data)
        assert isinstance(result, np.ndarray)
        # Label column should have been dropped, so width should match gene count
        # The raw df has 200 genes + 1 label; after drop we expect 200 cols
        assert result.shape[1] == 200

    def test_shape_consistency(self, gene_expression_csv):
        loader = GeneExpressionDataLoader(gene_expression_csv, preprocess_mode="log_zscore")
        data = loader.load_data()
        result = loader.preprocess_data(data)
        # Number of samples should be preserved
        assert result.shape[0] == data.shape[0]


# ======================================================================
#   Coordinate / polygon helpers
# ======================================================================

class TestCoordinateHelpers:

    def test_extract_coordinates_basic(self):
        assert extract_coordinates("tile_(100,200).png") == (100, 200)

    def test_extract_coordinates_no_match(self):
        assert extract_coordinates("random_file.png") is None

    def test_parse_tile_coords_with_floats(self):
        result = parse_tile_coords("tile_(14602.488, 1537.104).png")
        assert result is not None
        x, y = result
        assert abs(x - 14602.488) < 1e-3
        assert abs(y - 1537.104) < 1e-3

    def test_parse_tile_coords_negative(self):
        result = parse_tile_coords("tile_(-100.5, -200.3).png")
        assert result is not None
        assert result[0] < 0 and result[1] < 0

    def test_parse_tile_coords_no_match(self):
        assert parse_tile_coords("no_coords_here.png") is None


class TestCreatePolygons:

    def test_from_dataframe(self):
        df = pd.DataFrame({"coord_x": [0, 512], "coord_y": [0, 512]})
        polys = create_polygons(df)
        assert len(polys) == 2
        # Each polygon should be a 512×512 square
        for p in polys:
            assert abs(p.area - 512 * 512) < 1e-6

    def test_from_filenames_basic(self):
        fnames = ["tile_(0, 0).png", "tile_(512, 0).png"]
        polys = create_polygons_from_filenames(fnames, tile_size_px=512.0)
        assert len(polys) == 2
        for p in polys:
            assert abs(p.area - 512 * 512) < 1e-6

    def test_from_filenames_with_mpp_conversion(self):
        """If slide_mpp and target_mpp are provided, polygons should be rescaled."""
        fnames = ["tile_(1000.0, 2000.0).png"]
        polys_no_mpp = create_polygons_from_filenames(fnames, tile_size_px=512.0)
        polys_with_mpp = create_polygons_from_filenames(
            fnames, tile_size_px=512.0,
            slide_mpp=0.25, target_mpp=0.5,
            coords_in_microns=False,
        )
        # With slide_mpp=0.25 and target_mpp=0.5, the tile size in target
        # pixels should be 512 * (0.25/0.5) = 256 → area = 256² = 65536
        if polys_with_mpp:
            assert abs(polys_with_mpp[0].area - 256 * 256) < 1.0

    def test_from_filenames_empty_list(self):
        assert create_polygons_from_filenames([]) == []


# ======================================================================
#   Polygon validity helpers
# ======================================================================

class TestPolygonValidity:

    def test_contains_nan(self):
        from shapely.geometry import Polygon as ShapelyPolygon

        p = ShapelyPolygon([(0, 0), (1, 0), (1, float("nan")), (0, 1)])
        assert contains_nan_or_inf(p) is True

    def test_contains_inf(self):
        from shapely.geometry import Polygon as ShapelyPolygon

        p = ShapelyPolygon([(0, 0), (1, 0), (1, float("inf")), (0, 1)])
        assert contains_nan_or_inf(p) is True

    def test_valid_polygon_no_nan(self):
        from shapely.geometry import Polygon as ShapelyPolygon

        p = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert contains_nan_or_inf(p) is False

    def test_fix_invalid_removes_nan_coords(self):
        from shapely.geometry import Polygon as ShapelyPolygon

        p = ShapelyPolygon([(0, 0), (1, 0), (1, float("nan")), (0, 1)])
        fixed = fix_invalid_polygon(p)
        assert contains_nan_or_inf(fixed) is False


# ======================================================================
#   create_dataframe
# ======================================================================

class TestCreateDataframe:

    def test_returns_dataframe_without_augmented(self):
        coords = np.array([[0, 0], [512, 512]])
        feats = np.random.rand(2, 10)
        df = create_dataframe((None, coords, feats, None))
        assert isinstance(df, pd.DataFrame)
        assert "coord_x" in df.columns
        assert "coord_y" in df.columns
        assert df.shape[0] == 2

    def test_returns_dataframe_with_augmented(self):
        coords = np.array([[0, 0], [512, 512]])
        feats = np.random.rand(2, 10)
        aug = np.array([0, 1])
        df = create_dataframe((aug, coords, feats, None))
        assert isinstance(df, pd.DataFrame)

    def test_feature_columns_present(self):
        coords = np.array([[0, 0]])
        feats = np.random.rand(1, 5)
        df = create_dataframe((None, coords, feats, None))
        for i in range(5):
            assert f"feature_{i + 1}" in df.columns


# ======================================================================
#   ZIP / tiler_params helpers
# ======================================================================

class TestTilerParams:

    def test_read_tiler_params_valid_zip(self, tmp_path):
        """Should read tiler_params.json from inside a ZIP."""
        params = {"tile_size_um": 256.0, "tile_size_px": 512}
        zip_path = tmp_path / "slide.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("tiler_params.json", json.dumps(params))
        result = read_tiler_params(str(zip_path))
        assert result == params

    def test_read_tiler_params_missing_json(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        result = read_tiler_params(str(zip_path))
        assert result == {}

    def test_get_mpp_simple(self):
        params = {"tile_size_um": 256.0, "tile_size_px": 512}
        mpp = get_mpp_from_tiler_params(params)
        assert mpp is not None
        assert abs(mpp - 0.5) < 1e-9

    def test_get_mpp_zero_px(self):
        params = {"tile_size_um": 256.0, "tile_size_px": 0}
        assert get_mpp_from_tiler_params(params) is None

    def test_get_mpp_missing_keys(self):
        assert get_mpp_from_tiler_params({}) is None


class TestMatchZipToCsv:

    def test_exact_prefix_match(self):
        csv_list = [
            "TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF.csv",
            "TCGA-3C-AALJ-01Z-00-DX1.777C0957.csv",
        ]
        result = match_zip_to_csv("TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF.zip", csv_list)
        assert result == csv_list[0]

    def test_no_match_returns_none(self):
        csv_list = ["unrelated.csv"]
        result = match_zip_to_csv("TCGA-ZZ-0000.zip", csv_list, max_prefix=20)
        assert result is None


class TestReadZipTileList:

    def test_returns_only_tile_filenames(self, tmp_path):
        zip_path = tmp_path / "tiles.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("tile_(100.0, 200.0).png", b"fake_image")
            zf.writestr("tile_(300.0, 400.0).png", b"fake_image")
            zf.writestr("tiler_params.json", "{}")
        result = read_zip_tile_list(str(zip_path))
        assert len(result) == 2
        assert all("tile_" in n for n in result)


# ======================================================================
#   read_annotations
# ======================================================================

class TestReadAnnotations:

    def test_reads_csv_annotation(self, tmp_path):
        """Build a minimal annotation CSV and verify polygon reading."""
        csv_path = tmp_path / "anno.csv"
        lines = [
            "X_base,Y_base\n",
            "0,0\n",
            "100,0\n",
            "100,100\n",
            "0,100\n",
        ]
        csv_path.write_text("".join(lines))
        polys, rects = read_annotations(str(csv_path))
        # Should produce at least one polygon
        assert polys is not None
        assert rects.shape[0] >= 1

    def test_missing_columns_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("col_a,col_b\n1,2\n")
        with pytest.raises(IndexError):
            read_annotations(str(csv_path))
