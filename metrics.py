# -*- encoding: utf-8 -*-
#@File    :   metrics.py
#@Time    :   2022/10/16 20:41:56
#@Author  :   Li Suchi 
#@Email   :   lsuchi@126.com
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

'''
METRICS I/O
input: numpy.ndarray
output: metric result
'''
def MAE(y, y_pred):
    return np.mean(np.abs(y - y_pred))


def RMSE(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def RMSLE(y, y_pred):
    # RMSLE惩罚欠预测大于过预测
    log_minus = np.log(y_pred + 1) - np.log(y + 1)
    return np.sqrt(np.mean(log_minus ** 2))


def MAPE(y, y_pred):
    return np.mean(np.abs(y - y_pred) / y)


def WMAPE(y, y_pred):
    return np.sum(np.abs(y - y_pred)) / np.sum(y)


def SMAPE(y, y_pred):
    abs_minus = np.abs(y - y_pred)
    abs_mean = (np.abs(y) + np.abs(y_pred)) / 2
    return np.mean(abs_minus / abs_mean)


def MASK_MAPE(y, y_pred):
    pass


def AUC(y, y_pred):
    pass


def F1(y, y_pred):
    pass


def PRECISION(y, y_pred):
    pass


def RECALL(y, y_pred):
    pass