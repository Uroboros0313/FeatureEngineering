# -*- encoding: utf-8 -*-
#@File    :   fe_category.py
#@Time    :   2022/09/28 02:09:21
#@Author  :   Li Suchi 
#@Email   :   lsuchi@126.com
from itertools import combinations, product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from feature import BaseFeature


'''
CAT ENCODE
'''
class TargetEncode(BaseFeature):
    def __init__(self, params):
        self.cols = params.get("cols",None)
        self.label = params.get('label', None)
        self.if_prefix = params.get('if_prefix', False)
        self.maps = {}
    
    def fit(self, df):
        cols = self.cols

        for col in cols:
            dct = dict(df.groupby(col)[self.label].mean())
            self.maps[col] = dct

    def transform(self, df):
        cols = self.cols

        for col in cols:
            if self.if_prefix:
                df[f'TargetEnc_{col}'] = df[col].map(self.maps[col])
            else:
                df[col] = df[col].map(self.maps[col])

        return df


class LabelEncode(BaseFeature):
    def __init__(self, params):
        self.cols = params.get("cols", [])
        self.if_prefix = params.get('if_prefix', False)
        self.encoders = {}

    def fit(self, df):
        from sklearn.preprocessing import LabelEncoder

        for col in self.cols:
            encoder= LabelEncoder().fit(df[col])
            self.encoders[col] = encoder
    
    def transform(self, df):
        for col in self.cols:
            if self.if_prefix:
                df[f'LabelEnc_{col}'] = self.encoders[col].transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])

        return df
 
        
class CountEncode(BaseFeature):
    def __init__(self, params):
        self.cols = params.get('cols', [])
        self.if_prefix = params.get('if_prefix', False)
        self.map_dict = {}
    
    @staticmethod
    def value_cnt_map(ss: pd.Series):
        counts = ss.value_counts()
        return ss.name, counts
    
    def fit(self, df):
        op = self.value_cnt_map
        res = Parallel(n_jobs=-1, require='sharedmem')(
            delayed(op)(df[col]) for col in self.cols)
        
        self.map_dict.update(dict(res))

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            map_ = self.map_dict.get(col)
            if self.if_prefix:
                df[f'CntEnc_{col}'] = df[col].map(map_)
            else:
                df[col] = df[col].map(map_)

        return df


class OneHotEncode(BaseFeature):
    pass


class WoeEncode(BaseFeature):
    pass


class IVEncode(BaseFeature):
    pass


'''
CAT FEATURE
'''
class CatEqual(BaseFeature):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)
        self.prefix = params.get("prefix", "CatEqual_")
    
    def fit(self, df):
        pass

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"{self.prefix}({col1})({col2})"] = (df[col1]==df[col2]).astype(int)

        return df


class CatCross(BaseFeature):
    def __init__(self, params):
        self.cols = params.get("cols",None)
        self.prefix = params.get("prefix", "CatCross_")
        self.combs = None
        self.maps = {}

    def fit(self, df):
        combs = list(combinations(self.cols, 2))
        self.combs = combs
        for comb in combs:
            val_combs = list(product(df[comb[0]].unique(), df[comb[1]].unique()))
            val_codes = list(range(len(val_combs)))
            dct = dict(zip(val_combs, val_codes))
            self.maps[f'{comb[0]}_{comb[1]}'] = dct
    
    def transform(self, df):
        combs = self.combs
        for comb in combs:
            dct = self.maps.get(f'{comb[0]}_{comb[1]}')
            df[f'{self.prefix}{comb[0]}_{comb[1]}'] = pd.Series(zip(df[comb[0]], df[comb[1]])).map(dct)
        return df


class GroupStats(BaseFeature):
    def __init__(self, params):
        self.col_pairs = params.get('col_pairs', [])
        self.method = params.get('method', 'mean')
        self.map_dicts = {}

    def fit(self,df):
        for col1, col2 in self.col_pairs:
            series = df.groupby([col1])[col2].aggregate(self.method)
            self.map_dicts[f'{col1}_{col2}'] = series
    
    def transform(self, df):
        for col1, col2 in self.col_pairs:
            series = self.map_dicts.get(f'{col1}_{col2}')
            df[f'Group_{self.method}_({col1})({col2})'] = df[f'{col1}'].map(series).values

        return df
