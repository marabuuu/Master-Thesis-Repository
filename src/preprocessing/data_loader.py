import pandas as pd
import scanpy as sc

class GeneExpressionDataLoader:
    def __init__(self, csv_file, columns_to_drop=None):
        self.csv_file = csv_file
        self.columns_to_drop = columns_to_drop

    def load_data(self):
        data = pd.read_csv(self.csv_file, index_col=0)
        return data

    def preprocess_data(self, data):
        if self.columns_to_drop is not None:
            data = data.drop(columns=self.columns_to_drop)
        scaled_data = sc.pp.scale(data.values)
        return scaled_data