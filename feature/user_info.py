# coding:utf-8


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt



# 用户id,性别,职业,教育程度,婚姻状态,户口类型
names = ['userid','sex','job','edu','marriage','account']
user_info_train = pd.read_csv("../data/user_info_train.txt",header=None)
user_info_test = pd.read_csv("../data/user_info_test.txt",header=None)
user_info_train.columns = names
user_info_test.columns = names
user_info = pd.concat([user_info_train,user_info_test],ignore_index=True)

user_info['un0'] = (user_info==0).sum(axis=1)

user_info.loc[user_info.sex==0,'sex']=-9999
user_info.loc[user_info.account==0,'account']=-9999
user_info.loc[(user_info.job==0),'job']=-9999
user_info.loc[(user_info.job==1),'job']=-9999

user_info.loc[(user_info.edu==0) ,'edu']=-9999
user_info.loc[(user_info.edu==1) ,'edu']=-9999

user_info.loc[(user_info.marriage==5) ,'marriage']=-9999
user_info.loc[(user_info.marriage==0),'marriage']=-9999

user_info.loc[user_info.marriage==3 ,'marriage']=0

# one hot encode

for c in ['account','job','edu','marriage']:
    us = list(user_info[c].unique())
    us = [i for i in us if i!=-9999]
    for k in us:
        user_info[c+str(k)]=0
        user_info.loc[user_info[c]==k,c+str(k)]=1
user_info.set_index('userid',inplace=True)
user_info.sort_index(inplace=True)

user_info.to_csv('../data/user_data_dummy.csv')















