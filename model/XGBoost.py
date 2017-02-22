# coding:utf-8


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
#from xgboost import XGBClassifier as xgb
from sklearn import metrics


def getDatas(dir=''):
    loan_data = pd.read_csv("../data/{}.csv".format(dir))

    loan_data = loan_data.fillna(-9999)
    loan_data.index = loan_data['userid']
    loan_data.drop('userid',axis=1,inplace=True)

    target = pd.read_csv('../data/train/overdue_train.txt',
                         header=None)
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop('userid',axis=1,inplace=True)
    train = loan_data.iloc[0: 55596, :]
    test = loan_data.iloc[55596:, :]
    del loan_data
    return train,target,test


train,target,test = getDatas()
dtrain = xgb.DMatrix(train, label=target, missing=-9999)
dtest = xgb.DMatrix(test,missing=-9999)

watchlist = [(dtrain, 'train')]

param = {'booster':'gbtree',
			 'objective': 'binary:logistic',
			 'eval_metric':'auc',
			 'gamma': 0.1,
			 'max_depth': 6,
			 'lambda': 160,
			 'subsample': 0.8,
			 'colsample_bytree': 0.7,
			 'colsample_bylevel': 0.6,
			 'eta': 0.12,
			 'tree_method': 'exact',
			 'seed': 0,
			 'nthread': 12
			 }

bst = xgb.train(param, dtrain, 3500, watchlist, early_stopping_rounds=50)
dtest = xgb.DMatrix(test, missing=-9999)

preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)


