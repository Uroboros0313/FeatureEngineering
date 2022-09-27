# -*- encoding: utf-8 -*-
#@File    :   dev_models.py
#@Time    :   2022/09/28 02:08:42
#@Author  :   Li Suchi 
#@Email   :   lsuchi@126.com
import os
import pickle as pkl

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, XGBRanker
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.metrics import roc_auc_score

SEED = 2022

class BaseModel():
    # TODO:implements BaseModel super class
    pass


# TODO: 实现lightgbm, catboost, xgboost的原生接口代码
# TODO: 实现scikit-learn其余模型的接入
# TODO: 实现并解耦time-series模型
# models: LSTM, GRU, TCN, WaveNet, DeepAR, TFT, N-BEATS, Prophet, NeuralProphet
# packages: gluonts, torch_forecasting, kats, darts

class LGBMModel():
    def __init__(self, params={}):
        self.task_type = params.get('task_type', 'binary')
        self.model_params = params.get('model_params', {})
        defaults = {
            'objective':'binary',
            'boosting_type':'gbdt',
            'num_leaves':2**6,
            'max_depth':7,
            'n_jobs':-1,
            'colsample_bytree':0.8,
            'subsample_freq':1,
            'max_bin':255,
            'learning_rate':0.05, 
            'n_estimators':100, 
            'random_state':SEED,
            'verbose':-1,
        }
        
        if self.task_type in ['binary', 'multiclass']:
            defaults['objective'] = self.task_type
            defaults['metric'] = ['auc']
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = LGBMClassifier(**self.model_params)

        elif self.task_type == 'regression':
            defaults['objective'] = self.task_type
            defaults['metric'] = ['rmse', 'mape', 'mae']
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = LGBMRegressor(**self.model_params)
        
        elif self.task_type == 'rank':
            defaults['objective'] = 'lambdarank'
            defaults['metric'] = ['ndcg']
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = LGBMRanker(**self.model_params)

        else:
            raise ValueError("task_type error")

    def fit(self, trn_x, trn_y, val_x, val_y):
        fit_params = {
            'eval_set':[(trn_x, trn_y), (val_x, val_y)],
            'eval_names':['train', 'val'],
            'callbacks':[lgb.early_stopping(50), lgb.log_evaluation(10)]
            }

        print('-' * 100)
        print('start fitting LGBM model...')
        print('-' * 100)
        self.model.fit(trn_x, trn_y, **fit_params)     
        
        if self.task_type == 'binary':
            pred_y = self.model.predict_proba(val_x)[:,1]           
            print("LGBM model auc score: {}".format(roc_auc_score(val_y, pred_y)))
        
        return self.model

    def predict(self, test_x):
        preds = self.model.predict(test_x)
        return preds
    
    def predict_proba(self, test_x):
        if isinstance(self.model, LGBMClassifier):
            probs = self.model.predict_proba(test_x)
            return probs
        return 

    def save_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        with open(f'{model_path}.pkl', 'wb') as f:
            pkl.dump(self.model, f)

    def load_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        with open(f'{model_path}.pkl', 'rb') as f:
            self.model = pkl.load(f)
        return self.model


class XGBModel():
    def __init__(self, params={}):
        self.task_type = params.get('task_type', 'binary')
        self.model_params = params.get('model_params', {})
        defaults = {
            'max_depth':6,
            'n_jobs':-1,
            'learning_rate':0.1, 
            'n_estimators':200, 
            'random_state':SEED,
            'verbosity':1,
            'early_stopping_rounds':20,
            'callbacks':[xgb.callback.EvaluationMonitor(period=50)]
        }
        if self.task_type == 'binary':
            defaults['objective'] = 'binary:logistic'
            defaults['eval_metric'] = ["auc"]
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = XGBClassifier(**self.model_params)

        elif self.task_type == 'multiclass':
            defaults['objective'] = 'multi:softprob'
            defaults['eval_metric'] = ["auc"]
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = XGBClassifier(**self.model_params)
        
        elif self.task_type == 'regression':
            defaults['objective'] = 'reg:squarederror'
            defaults['eval_metric'] = ["rmse", "mae", "mape"]
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = XGBRegressor(**self.model_params)

        elif self.task_type == 'rank':
            defaults['objective'] = 'rank:pairwise'
            defaults['eval_metric'] = ["aucpr"]
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = XGBRanker(**self.model_params)

    def fit(self, trn_x, trn_y, val_x, val_y):
        fit_params = {
            'eval_set':[(trn_x, trn_y), (val_x, val_y)],
            }

        print('-' * 100)
        print('start fitting XGBoost model...')
        print('-' * 100)
        self.model.fit(trn_x, trn_y, **fit_params)
        
        if self.task_type == 'binary':
            pred_y = self.model.predict_proba(val_x)[:,1]           
            print("XGB model auc score: {}".format(roc_auc_score(val_y, pred_y)))
        
        return self.model

    def predict(self, test_x):
        preds = self.model.predict(test_x)
        return preds
    
    def predict_proba(self, test_x):
        if isinstance(self.model, XGBClassifier):
            probs = self.model.predict_proba(test_x)
            return probs
        return

    def save_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        with open(f'{model_path}.pkl', 'wb') as f:
            pkl.dump(self.model, f)
    
    def load_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        with open(f'{model_path}.pkl', 'rb') as f:
            self.model = pkl.load(f)
        return self.model


class CatBoostModel():
    def __init__(self, params={}):
        self.task_type = params.get('task_type', 'binary')
        self.model_params = params.get('model_params', {})
        self.cat_features = params.get('cat_features', None)

        defaults = {
            "iterations":210, 
            "depth":6, 
            "learning_rate":0.03, 
            "l2_leaf_reg":1,  
            #"verbose":0,
            "metric_period":50,
            "cat_features":self.cat_features,
            "random_seed":SEED,
        }

        if self.task_type == 'binary':
            defaults['eval_metric'] = "AUC"
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = CatBoostClassifier(**self.model_params)
        
        elif self.task_type == 'multiclass':
            defaults['eval_metric'] = "AUC"
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = CatBoostClassifier(**self.model_params)

        elif self.task_type == 'regression':
            defaults['eval_metric'] = "MAPE"
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = CatBoostRegressor(**self.model_params)

        elif self.task_type == 'rank':
            defaults['eval_metric'] = "NDCG"
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = CatBoostRanker(**self.model_params)

    def fit(self, trn_x, trn_y, val_x, val_y):
        fit_params = {
            "eval_set":[(trn_x, trn_y), (val_x, val_y)],
            "early_stopping_rounds":50
        }

        print('-' * 100)
        print('start fitting CatBoost model...')
        print('-' * 100)
        self.model.fit(trn_x, trn_y, **fit_params)
        
        if self.task_type == 'binary':
            pred_y = self.model.predict_proba(val_x)[:,1]           
            print("CGB model auc score: {}".format(roc_auc_score(val_y, pred_y)))
        
        return self.model

    def predict(self, test_x):
        preds = self.model.predict(test_x)
        return preds
    
    def predict_proba(self, test_x):
        if isinstance(self.model, CatBoostClassifier):
            probs = self.model.predict_proba(test_x)
            return probs
        
        return 

    def save_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        with open(f'{model_path}.pkl', 'wb') as f:
            pkl.dump(self.model, f)

    def load_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        with open(f'{model_path}.pkl', 'rb') as f:
            self.model = pkl.load(f)
        return self.model


###########################################################################

class TabNetModel():
    
    def __init__(self, params={}):
        self.task_type = params.get('task_type', 'binary')
        self.model_params = params.get('model_params', {})
        defaults = {
            'optimizer_fn':torch.optim.Adam,
            'optimizer_params':dict(lr=2e-2),
            'scheduler_params':{
                "step_size":10, # how to use learning rate scheduler
                "gamma":0.9
                },
            'scheduler_fn':torch.optim.lr_scheduler.StepLR,
            'mask_type':'sparsemax', # This will be overwritten if using pretrain model
            'seed':SEED,
            'verbose':20,
        }

        if self.task_type == 'binary':
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = TabNetClassifier(**self.model_params)
        
        elif self.task_type == 'multiclass':
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = TabNetClassifier(**self.model_params)

        elif self.task_type == 'regression':
            defaults.update(self.model_params)
            self.model_params = defaults
            self.model = TabNetRegressor(**self.model_params)

    @staticmethod
    def assert_class(data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data = data.values
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            pass
        else:
            raise ValueError('tabnet `fit` input should be `list`, `np.ndarray`, `pd.DataFrame` or `pd.Series`')
        return data
    
    @staticmethod
    def revise_shape(data):
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], -1)
        return data
    
    def fit(self, trn_x, trn_y, val_x, val_y):
        trn_x, trn_y = self.assert_class(trn_x), self.assert_class(trn_y)
        val_x, val_y = self.assert_class(val_x), self.assert_class(val_y)
        
        if isinstance(self.model, TabNetRegressor):
            trn_x, trn_y = self.revise_shape(trn_x), self.revise_shape(trn_y)
            val_x, val_y = self.revise_shape(val_x), self.revise_shape(val_y)
        
        fit_params = {
            'max_epochs':100,
            'patience':10,
            'eval_set':[(trn_x, trn_y), (val_x, val_y)],
            'eval_name':['train', 'val'],
        }

        if self.task_type == 'binary':
            fit_params['eval_metric'] = ['accuracy', 'auc']
        elif self.task_type == 'multiclass':
            fit_params['eval_metric'] = ['accuracy']
        elif self.task_type == 'regression':
            fit_params['eval_metric'] = ['rmse', 'mae']

        print('-' * 100)
        print('start fitting TabNet model...')
        print('-' * 100)
        self.model.fit(trn_x, trn_y, **fit_params) 
        if self.task_type == 'binary':
            pred_y = self.model.predict_proba(val_x)[:,1]         
            print("TabNet model auc score: {}".format(roc_auc_score(val_y, pred_y)))
        
        return self.model

    def predict(self, test_x):
        test_x = self.assert_class(test_x)
        preds = self.model.predict(test_x)
        return preds
    
    def predict_proba(self, test_x):
        test_x = self.assert_class(test_x)
        if isinstance(self.model, TabNetClassifier):
            probs = self.model.predict_proba(test_x)
            return probs
        return

    def save_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        saved_filepath = self.model.save_model(model_path)
        return saved_filepath

    def load_model(self, model_dir, model_name):
        model_path = os.path.join(model_dir, model_name)
        self.model.load_model(f"{model_path}.zip")
        return self.model


        

                

        





        