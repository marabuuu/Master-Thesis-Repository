import pandas as pd
import numpy as np
from src.preprocessing.utils import preprocess_log1p_zscore, preprocess_log1p_minmax, inspect_variance


class GeneExpressionDataLoader:
    def __init__(self, csv_file, columns_to_drop=None, id_column=None, preprocess_mode='auto'):
        """
        csv_file: path to expression CSV
        columns_to_drop: list of feature columns to remove (ignored if missing)
        id_column: optional column name to use as sample id/index
        preprocess_mode: 'auto' | 'log_zscore' | 'log_minmax' | 'none'
        """
        self.csv_file = csv_file
        self.columns_to_drop = columns_to_drop or []
        self.id_column = id_column
        self.preprocess_mode = preprocess_mode

    def load_data(self):
        data = pd.read_csv(self.csv_file)
        if self.id_column and self.id_column in data.columns:
            data = data.set_index(self.id_column)
        else:
            # fallback: first column becomes index
            data = data.set_index(data.columns[0])
        return data

    def preprocess_data(self, data):
        """Preprocess data according to `preprocess_mode`.

        If `preprocess_mode` is 'auto', detect whether data already appears
        standardized (mean ~0, std ~1) and skip log1p if so. Otherwise apply
        log1p + z-score.
        Returns NumPy array (samples x genes) ready for torch.
        """
        if self.columns_to_drop:
            data = data.drop(columns=self.columns_to_drop, errors="ignore")

        # Basic detection: check gene mean/std
        stats = inspect_variance(data)
        gene_mean = np.mean([stats['gene_var_summary'].get('mean', 0)])
        gene_median = stats['gene_var_summary'].get('50%', None)

        if self.preprocess_mode == 'none':
            return data.values.astype(float)

        if self.preprocess_mode == 'auto':
            # If medians/std indicate z-scored inputs (median near 1 for var and mean near 0), skip log1p
            # Here we check if gene variance median is between 0.5 and 1.5 as heuristic
            gene_var_median = stats['gene_var_summary'].get('50%', None)
            if gene_var_median is not None and 0.5 <= gene_var_median <= 1.5:
                return data.values.astype(float)
            else:
                # assume raw counts
                return preprocess_log1p_zscore(data).values.astype(float)

        if self.preprocess_mode == 'log_zscore':
            return preprocess_log1p_zscore(data).values.astype(float)

        if self.preprocess_mode == 'log_minmax':
            return preprocess_log1p_minmax(data).values.astype(float)

        # fallback
        return preprocess_log1p_zscore(data).values.astype(float)