import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_log1p_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p then per-gene min-max scaling to [0,1].

    Useful when decoder uses sigmoid and reconstruction target is expected in [0,1].
    """
    arr = np.log1p(df.values.astype(float))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def preprocess_log1p_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p then z-score per gene (zero mean, unit var).

    Useful when decoder is identity and reconstruction uses MSE.
    """
    arr = np.log1p(df.values.astype(float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def inspect_variance(df: pd.DataFrame) -> dict:
    """Return simple variance diagnostics for the dataframe.
    """
    gene_var = df.var(axis=0)
    sample_var = df.var(axis=1)
    return {
        'n_genes': df.shape[1],
        'n_samples': df.shape[0],
        'gene_var_summary': gene_var.describe().to_dict(),
        'n_zero_var_genes': int((gene_var == 0).sum()),
        'sample_var_summary': sample_var.describe().to_dict()
    }
