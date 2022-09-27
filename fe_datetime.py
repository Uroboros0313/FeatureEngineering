import numpy as np

from feature import BaseFeature


# DATE FEATURE
class DateTimeSplit(BaseFeature):
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


class KeyTimeShift(BaseFeature):
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


class KeyTimeDiff(BaseFeature):
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


class TimeMinus2(BaseFeature):
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
