import pandas as pd
import matplotlib.pyplot as plt

SEED=2022


class DataSet:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.train_ids = None
        self.test_x = None
        self.test_ids = None
        self.n_splits = None

    def set_train(self, train_x, train_y ,train_ids):
        self.train_x = train_x
        self.train_y = train_y
        self.train_ids = train_ids

    def set_test(self, test_x, test_ids):
        self.train_x = test_x
        self.train_ids = test_ids

    def kfold_split(self, n_splits=3):
        from sklearn.model_selection import StratifiedKFold
        self.n_splits = n_splits
        kf = StratifiedKFold(
            n_splits=n_splits, 
            random_state=SEED, 
            shuffle=True
            )
        return kf.split(self.train_x, self.train_y)

    def train_test_split(self, test_size=0.3):
        from sklearn.model_selection import train_test_split
        trn_x, val_x, trn_y, val_y =\
             train_test_split(self.train_x, self.train_y, test_size=test_size)

        return trn_x, val_x, trn_y, val_y


class DataCenter:
    def __init__(self):
        self.train_idxs = None
        self.test_idxs = None
        self.nolabel_idxs = None
        self.raw_feats = None
        self.all_data = None
        self.label = None
        self.id = None
        self.dataset = None

    def prepare_dataset(self, del_cols=[]):
        del_cols = del_cols + [self.label, self.id]
        keep_cols = self.test.columns.to_list()
        keep_cols = [col for col in keep_cols if col not in del_cols]

        train_data = self.train.copy()
        train_y = train_data[self.label]
        train_ids = train_data[self.id]
        train_x = train_data[keep_cols]

        test_data = self.test.copy()
        test_ids = test_data[self.id]
        test_x = test_data[keep_cols]

        dataset = DataSet()
        dataset.set_train(train_x, train_y, train_ids)
        dataset.set_test(test_x, test_ids)
        self.dataset = dataset
        return dataset
    
    def load_excel(self, path, name):
        df = pd.read_excel(path)
        setattr(self, name, df)

    def load_csv(self, path, name, encode='utf-8'):
        df = pd.read_csv(path, encoding=encode)
        setattr(self, name, df)

    def show_cat_cols(self, name):
        df = getattr(self, name)
        self.cat_cols = df.select_dtypes(include=object).columns.to_list()
        print('dataset:{}, categorical columns:{}'.format(name, self.cat_cols))
        print("-"*100)
    
    def show_int_cols(self, name):
        df = getattr(self, name)
        self.int_cols = df.select_dtypes(include=int).columns.to_list()
        print('dataset:{}, int columns:{}'.format(name, self.int_cols))
        print("-"*100)

    def show_float_cols(self, name):
        df = getattr(self, name)
        self.float_cols = df.select_dtypes(include=float).columns.to_list()
        print('dataset:{}, float columns:{}'.format(name, self.float_cols))
        print("-"*100)

    def show_binary_cols(self, name):
        df = getattr(self, name)
        columns = df.columns.to_list()
        self.binary_cols = [col for col in columns if df[col].nunique()==2]
        print('dataset:{}, binary columns:{}'.format(name, self.binary_cols))
        print("-"*100)

    def show_alltype_cols(self, name):
        self.show_binary_cols(name)
        self.show_int_cols(name)
        self.show_cat_cols(name)
        self.show_float_cols(name)


def anlys_imblance(series: pd.Series):
    grp_vals = series.groupby(series).count()
    plt.bar(grp_vals.index.astype(str), grp_vals.values)
    plt.show()