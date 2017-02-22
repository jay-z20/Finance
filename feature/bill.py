# coding:utf-8


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt



bill_data = pd.read_csv('../data/bill_no_duplicate.csv')

bill_data.loc[:, 'repay_sub'] = bill_data.loc[:, 'pre_amount_of_bill'] - bill_data.loc[:, 'pre_repayment']
bill_data.loc[:, 'repay_sub_now'] = bill_data.loc[:, 'amount_of_bill'] - bill_data.loc[:, 'pre_repayment']

names_loan_time = ['userid','loan_time']
loan_time_train = pd.read_csv("../../pcredit/train/loan_time_train.txt",header=None)
loan_time_test = pd.read_csv("../../pcredit/test/loan_time_test.txt",header=None)
loan_time = pd.concat([loan_time_train,loan_time_test],axis=0)
loan_time.columns = names_loan_time
del loan_time_train,loan_time_test

bill_data = pd.merge(bill_data,loan_time,on='userid',how='left')
del loan_time,names_loan_time

users = list(bill_data.userid.unique())


cols = ['repay_sub','repay_sub_now','pre_amount_of_bill','pre_repayment','consume_amount','credit_amount','amount_of_bill_left']


def bill_time_split2(user):

    d = {'userid': user}
    bills = bill_data[bill_data.userid == user]

    for col in cols:
        t = bills[col]
        d[col + '_min'] = t.min()
        d[col + '_max'] = t.max()
        d[col + '_median'] = t.median()
        d[col + '_mean'] = t.mean()
        d[col + '_std'] = t.std()
        d[col + '_max_min'] = t.max() - t.min()

        t = bills[bills.time <= bills.loan_time][col]
        d[col + '_min_lt'] = t.min()
        d[col + '_max_lt'] = t.max()
        d[col + '_median_lt'] = t.median()
        d[col + '_mean_lt'] = t.mean()
        d[col + '_std_lt'] = t.std()
        d[col + '_max_min_lt'] = t.max() - t.min()

        t = bills[bills.time > bills.loan_time][col]
        d[col + '_min_gt'] = t.min()
        d[col + '_max_gt'] = t.max()
        d[col + '_median_gt'] = t.median()
        d[col + '_mean_gt'] = t.mean()
        d[col + '_std_gt'] = t.std()
        d[col + '_max_min_gt'] = t.max() - t.min()

        d[col + '_min_gt_lt'] = d[col + '_min_gt'] - d[col + '_min_lt']
        d[col + '_max_gt_lt'] = d[col + '_max_gt'] - d[col + '_max_lt']
        d[col + '_mean_gt_lt'] = d[col + '_mean_gt'] - d[col + '_mean_lt']
        d[col + '_median_gt_lt'] = d[col + '_median_gt'] - d[col + '_median_lt']
        d[col + '_max_min_gt_lt'] = d[col + '_max_min_gt'] - d[col + '_max_min_lt']
    print d
    return d


split_point_col = {}
stage = ['stg'+str(i)+"_" for i in range(1,11)]
for col in cols:
    t = bill_data[col].describe(percentiles=[0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9])
    split_point_col[col] = [-1e11,int(t['10%']), int(t['20%']),int(t['30%']), int(t['40%']),int(t['50%']),
                            int(t['60%']),int(t['70%']), int(t['80%']),int(t['90%']), 1e11]

def bill_data_bin_10(u):
    d = {'userid':u}
    data = bill_data[bill_data.userid == u]
    for col in cols:
        for i in range(10):
            stg = stage[i]
            di = data[(split_point_col[col][i] < (data[col])) & ((data[col]) < split_point_col[col][i + 1])]
            d[stg + col + '_cnt'] = di[col].count()
            d[stg + col + '_min'] = di[col].min()
            d[stg + col + '_max'] = di[col].max()
            d[stg + col + '_mean'] = di[col].mean()
            if col not in ['time']:
                d[stg + col + '_std'] = di[col].std()
    print d
    return d


def multi():
    from multiprocessing import Pool
    pool = Pool(8)

    for fun in [bill_time_split2,bill_data_bin_10]:
        rst = pool.map(fun,users)
        pool.close()
        pool.join()
        features = pd.DataFrame(rst)
        fileName = '../data/{}.csv'.format(fun.__name__)
        features.to_csv(fileName,index=None)

if __name__=='__main__':
    multi()











