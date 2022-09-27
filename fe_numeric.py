import numpy as np
import pandas as pd
from scipy import stats

from feature import BaseFeature


# NUMERIC TRANSFORM
class Bining(BaseFeature):
    def __init__(self, params):
        self.num_bins = params.get('num_bins', 5)
        self.cols = params.get('cols', [])
        self.bin_method = params.get('bin_method', 'width') # [width, quantile]
        self.include_offrange = params.get('include_offrange', True)
        self.bins = {}
    
    def fit(self, df):

        for col in self.cols:
            if self.bin_method == 'width':
                _, self.bins[col] = pd.cut(df[col], self.num_bins, retbins=True, labels=False)
            elif self.bin_method == 'quantile':
                _, self.bins[col] = pd.qcut(df[col], self.num_bins, retbins=True, labels=False)
            else:
                raise ValueError(f"Unknown bining method {self.bin_method}")
            
            if self.include_offrange:
                self.bins[col][-1] = np.inf
                self.bins[col][0] = -np.inf
            
    def transform(self, df):
        
        for col in self.cols:
            df[col] = pd.cut(df[col], self.bins[col], labels=False, include_lowest=True)
        
        return df


class BoxCoxTransform(BaseFeature):
    def __init__(self, params):
        self.cols = params.get('cols', [])
        self.lmbds = {}
        self.col_mins = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.cols:
            col_vals = df[col]
            col_min = col_vals.min()
            self.col_mins[col] = col_min
            if col_min <= 0:
                col_vals = col_vals - col_min + 1
            
            _, lmbd = stats.boxcox(col_vals)
            self.lmbds[col] = lmbd
    
    def transform(self, df: pd.DataFrame):
        for col in self.cols:
            if self.col_mins[col] <= 0:
                df[col] = stats.boxcox(df[col] - self.col_mins[col] + 1, lmbda=self.lmbds[col])
            elif self.col_mins[col] > 0:
                df[col] = stats.boxcox(df[col], lmbda=self.lmbds[col])
        return df


class LogTransform(BaseFeature):
    def __init__(self, params):
        self.cols = params.get("cols",None)

    def fit(self,df):
        pass
    
    def transform(self, df):
        for col in self.cols:
            df[col] = np.log(df[col] + 1)

        return df


class NAdd(BaseFeature):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NAdd_({col1})({col2})"] = df[col1]+df[col2]

        return df


class NMinus(BaseFeature):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NMinus_({col1})({col2})"] = df[col1]-df[col2]

        return df


class NMul(BaseFeature):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NMul_({col1})({col2})"] = df[col1]*df[col2]

        return df


class NDiv(BaseFeature):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)
        self.offset = params.get("offset",1)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NDiv_({col1})({col2})"] = df[col1]/(df[col2]+self.offset)
            df[f"NDiv_({col2})({col1})"] = df[col2]/(df[col1]+self.offset)

        return df


