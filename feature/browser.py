# coding:utf-8


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt





names = ['userid', 'time', 'browser_behavior', 'browser_behavior_number']
browse_history_train = pd.read_csv("../data/browse_history_train.txt", header=None)
browse_history_test = pd.read_csv("../data/browse_history_test.txt", header=None)

browse_history = pd.concat([browse_history_train, browse_history_test])
browse_history.columns = names
browse_history['browse_count'] = 1
del browse_history_train, browse_history_test

names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../data/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../data/loan_time_test.txt",header=None)
loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)
loan_time.columns = names_loan_time

browse_history = pd.merge(browse_history,loan_time,on='userid')

del loan_time


def browser_data_split_time(u):
    d = {'userid': u}
    du = browse_history[browse_history.userid == u]
    data = du[du.time <= du.loan_time]
    brdata = data[['userid', 'browser_behavior', 'browse_count']].groupby(['userid', 'browser_behavior']).agg(sum)
    brdata.reset_index(inplace=True)
    d['browse_data' + '_min'+'_lt'] = brdata['browse_count'].min()
    d['browse_data' + '_max'+'_lt'] = brdata['browse_count'].max()
    d['browse_data' + '_mean'+'_lt'] = brdata['browse_count'].mean()
    d['browse_data' + '_median'+'_lt'] = brdata['browse_count'].median()
    d['browse_data' + '_std'+'_lt'] = brdata['browse_count'].std()
    d['browse_data' + '_count'+'_lt'] = brdata['browse_count'].count()
    d['browse_data' + '_var'+'_lt'] = brdata['browse_count'].var()
    d['browse_data_log_cnt'+'_lt'] = brdata['browse_count'].sum()
    d['browse_data_max_min_lt'] = brdata['browse_count'].max() - brdata['browse_count'].min()
    del brdata,data

    data = du[du.time > du.loan_time]
    brdata = data[['userid', 'browser_behavior', 'browse_count']].groupby(['userid', 'browser_behavior']).agg(sum)
    brdata.reset_index(inplace=True)
    d['browse_data' + '_min'+'_gt'] = brdata['browse_count'].min()
    d['browse_data' + '_max'+'_gt'] = brdata['browse_count'].max()
    d['browse_data' + '_mean'+'_gt'] = brdata['browse_count'].mean()
    d['browse_data' + '_median'+'_gt'] = brdata['browse_count'].median()
    d['browse_data' + '_std'+'_gt'] = brdata['browse_count'].std()
    d['browse_data' + '_count'+'_gt'] = brdata['browse_count'].count()
    d['browse_data' + '_var'+'_gt'] = brdata['browse_count'].var()
    d['browse_data_log_cnt'+'_gt'] = brdata['browse_count'].sum()
    d['browse_data_max_min_gt'] = brdata['browse_count'].max() - brdata['browse_count'].min()

    d['browse_data_min_gt_lt'] = d['browse_data' + '_min'+'_gt'] - d['browse_data' + '_min'+'_lt']
    d['browse_data_max_gt_lt'] = d['browse_data' + '_max' + '_gt'] - d['browse_data' + '_max' + '_lt']
    d['browse_data_mean_gt_lt'] = d['browse_data' + '_mean' + '_gt'] - d['browse_data' + '_mean' + '_lt']
    d['browse_data_count_gt_lt'] = d['browse_data' + '_count' + '_gt'] - d['browse_data' + '_count' + '_lt']
    d['browse_data_log_gt_lt'] = d['browse_data_log_cnt'+'_gt'] - d['browse_data_log_cnt'+'_lt']
    del brdata, data,du
    print d
    return d



def multi_browser():

    from multiprocessing import Pool,Queue,Lock
    pool = Pool(4)
    users = list(browse_history.userid.unique())
    rst = pool.map(browser_data_split_time,users)

    features = pd.DataFrame(rst)
    features.fillna(-9999,inplace=True)

    features.to_csv('../data/browse_data_split_time.csv', index=None)




if __name__=="__main__":
    multi_browser()













