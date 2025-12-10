import pandas as pd
import scanpy as sc


class GeneExpressionDataLoader:
    def __init__(self, csv_file, columns_to_drop=None, id_column=None):
        """
        csv_file: path to expression CSV
        columns_to_drop: list of feature columns to remove (ignored if missing)
        id_column: optional column name to use as sample id/index
        """
        self.csv_file = csv_file
        self.columns_to_drop = columns_to_drop or []
        self.id_column = id_column

    def load_data(self):
        data = pd.read_csv(self.csv_file)
        if self.id_column and self.id_column in data.columns:
            data = data.set_index(self.id_column)
        else:
            # fallback: first column becomes index
            data = data.set_index(data.columns[0])
        return data

    def preprocess_data(self, data):
        if self.columns_to_drop:
            data = data.drop(columns=self.columns_to_drop, errors="ignore")
        scaled_data = sc.pp.scale(data.values)
        return scaled_data