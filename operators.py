from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed


class BaseOp(ABC):
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def transform(self):
        pass
'''
ROW FEATURE
'''
#DATE FEATURE
class DateTimeSplit(BaseOp):
    def __init__(self, params):
        self.time_cols = params.get('time_cols', None)
        self.time_parts = params.get('time_parts', ['day', 'week', 'month'])
        self.if_emb = params.get('if_emb', False)
        self.emb_map = None
        

    def fit(self, df):
        if self.if_emb:
            self.emb_map = {
                'day':31,
                'week':53,
                'weekday':7,
                'hour':24,
            }

    def transform(self, df):
        for col in self.time_cols:
            if 'day' in self.time_parts:
                df[f'day_{col}'] = df[col].dt.day
            if 'week' in self.time_parts:
                df[f'week_{col}'] = df[col].dt.week
            if 'weekday' in self.time_parts:
                df[f'weekday_{col}'] = df[col].dt.weekday
            if 'hour' in self.time_parts:
                df[f'hour_{col}'] = df[col].dt.hour

            if self.if_emb:
                for part in self.time_parts:
                    num = self.emb_map[part]
                    df[f'{part}_{col}_sin'] = np.sin(df[f'{part}_{col}']/num*np.pi)
                    df[f'{part}_{col}_cos'] = np.cos(df[f'{part}_{col}']/num*np.pi)
        return df


#NUMERIC TRANSFORM
class LogTransform(BaseOp):
    def __init__(self, params):
        self.cols = params.get("cols",None)

    def fit(self,df):
        pass
    
    def transform(self, df):
        for col in self.cols:
            df[col] = np.log(df[col] + 1)

        return df

class NAdd(BaseOp):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NAdd_({col1})({col2})"] = df[col1]+df[col2]

        return df

class NMinus(BaseOp):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NMinus_({col1})({col2})"] = df[col1]-df[col2]

        return df

class NMul(BaseOp):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)

    def fit(self, df):
        pass 

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"NMul_({col1})({col2})"] = df[col1]*df[col2]

        return df

class NDiv(BaseOp):
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

# CAT TRANSFORM
class CatEqual(BaseOp):
    def __init__(self, params):
        self.col_pairs = params.get("col_pairs",None)
        self.prefix = params.get("prefix", "CatEqual_")
    
    def fit(self, df):
        pass

    def transform(self, df):
        for col1, col2 in self.col_pairs:
            df[f"{self.prefix}({col1})({col2})"] = (df[col1]==df[col2]).astype(int)

        return df

class CatCross(BaseOp):
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

class KeyTimeShift(BaseOp):
    def __init__(self, params):
        self.user_col = params.get('user_col', None)
        self.key_time_col = params.get('key_time_col', None)
        self.shift_cols = params.get('shift_cols',None)
        self.shift_lens = params.get('diff_lens', [1,2])

    def fit(self,df):
        pass

    def transform(self, df):
        df = df.copy()
        df = df.sort_values(by = [self.user_col, self.key_time_col])

        for col in self.shift_cols:
            for i in self.shift_lens:
                df[f"{col}_shift_{i}"] = df.groupby(self.user_col)[col].shift(i)

        return df

class KeyTimeDiff(BaseOp):
    def __init__(self, params):
        self.user_col = params.get('user_col', None)
        self.key_time_col = params.get('key_time_col', None)
        self.shift_cols = params.get('shift_cols',None)
        self.diff_lens = params.get('diff_lens', [1,2,3])

    def fit(self,df):
        pass

    def transform(self, df):
        df = df.copy()
        df = df.sort_values(by = [self.user_col, self.key_time_col])
        for col in self.shift_cols:
            for i in self.diff_lens:
                df[f"{col}_Diff_{i}"] = df.groupby(self.user_col)[col].diff(i)

        return df

class TimeMinus2(BaseOp):
    def __init__(self, params):
        self.key_time_col = params.get('key_time_col')
        self.target_time_col = params.get('target_time_col')
        self.parts = params.get("parts",[])
        self.prefix = params.get('prefix', 'TIMEDIFF_')

    @staticmethod
    def gen_col_name(prefix, col_1, col_2):
        return f'{prefix}{col_1}_{col_2}'
    
    def fit(self, df = None):
        pass

    def transform(self, df):
        new_col_name = self.gen_col_name(self.prefix, self.key_time_col, self.target_time_col)
        df[new_col_name] = df[self.key_time_col] - df[self.target_time_col]
        if "days" in self.parts:
            df[f"{new_col_name}(Days)"] = df[new_col_name].dt.days
        if "hours" in self.parts:
            df[f"{new_col_name}(Hours)"] = df[new_col_name].dt.seconds/3600

        return df

'''
COL FEATURE
'''
# TODO: add Session Feature

class DeleteCols(BaseOp):
    def __init__(self, params):
        self.cols = params.get('cols', [])
    
    def fit(self, df):
        pass

    def transform(self, df):
        for col in self.cols:
            del df[col]

        return df

class GroupStats(BaseOp):
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
        
class GroupCnt(BaseOp):
    def __init__(self, params):
        self.cols = params.get('cols', [])
        self.prefix = params.get('prefix', 'GroupCnt_')
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
            df[f'{self.prefix}({col})'] = df[col].map(map_)

        return df

class KNearestLabels(BaseOp):
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

# Encode

class TargetEncode(BaseOp):
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
                df['TEncode_{}'.format(col)] = df[col].map(self.maps[col])
            else:
                df[col] = df[col].map(self.maps[col])

        return df

class LabelEncode(BaseOp):
    def __init__(self, params):
        self.cols = params.get("cols", [])
        self.encoders = {}

    def fit(self, df):
        from sklearn.preprocessing import LabelEncoder

        for col in self.cols:
            encoder= LabelEncoder().fit(df[col])
            self.encoders[col] = encoder
    
    def transform(self, df):
        for col in self.cols:
            df[col] = self.encoders[col].transform(df[col])

        return df

class MinMaxScaler(BaseOp):
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


