from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from feature import BaseFeature


# TODO: add Session Feature

class DeleteCols(BaseFeature):
    def __init__(self, params):
        self.cols = params.get('cols', [])
    
    def fit(self, df):
        pass

    def transform(self, df):
        for col in self.cols:
            del df[col]

        return df



class KNearestLabels(BaseFeature):
    def __init__(self, params):
        self.n_neighbors = params.get('n_neighbors', 3) + 1
        self.label = params.get('label', None)
        self.dis_metrics = params.get('dis_metrics', 'l1')
        self.id_col = params.get('id_col', None)
        self.self_mask_idxs = params.get('self_mask_idxs', [])
        self.source_labels = None
        self.select_cols = []
        self.nbrs = None

    def fit(self, df):
        source_df = df.copy()
        self.source_labels = source_df[self.label]
        del source_df[self.label]
        self.select_cols = [col for col in source_df.columns.to_list() if col != self.id_col]

        self.nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors, 
            algorithm='ball_tree', 
            metric=self.dis_metrics, 
            n_jobs=-1
            )
        self.nbrs.fit(source_df[self.select_cols])

    def transform(self, df):
        target_df = df.copy()
        if self.label in target_df.columns.to_list():
            del target_df[self.label]
        
        nb_idx = self.nbrs.kneighbors(target_df[self.select_cols], return_distance=False)# 60s内可以结束
        for i in range(1, self.n_neighbors):
            col_name = f'{self.dis_metrics}_{i}'
            if self.self_mask_idxs:
                label_series = pd.Series(self.source_labels.iloc[nb_idx[:, i - 1]].values)
                df[col_name] = label_series.values
                
                mask_series = pd.Series(self.source_labels.iloc[nb_idx[:, i]].values)
                df[col_name].loc[self.self_mask_idxs] =\
                     mask_series.loc[self.self_mask_idxs].values
            else:
                df[col_name] = self.source_labels.iloc[nb_idx[:, i]].values

        return df


class MinMaxScaler(BaseFeature):
    def __init__(self, params):
        self.dim = params.get("dim",2)
        self.id_col = params.get('id_col', None)
        self.label_col = params.get('label_col', None)
        self.select_cols = None
        self.max = None
        self.min = None

    def fit(self, df):
        self.select_cols = [col for col in df.columns if col not in [self.id_col, self.label_col]]

        tmp_df = df[self.select_cols]
        self.max = tmp_df.max()
        self.min = tmp_df.min()

    def transform(self, df):
        tmp_df = df[self.select_cols]
        tmp_df = (tmp_df - self.min) / (self.max - self.min)
        df[self.select_cols] = tmp_df

        return df

