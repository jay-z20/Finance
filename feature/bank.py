# coding:utf-8


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt

bank_data = pd.read_csv('../data/bank_no_duplicate.csv')


def bank_time_split_2(u):

    # 获得所有的用户
    d = {'userid': u}
    col = 'examount'
    bank_users = bank_data[bank_data.userid == u]
    for e in [0, 1]:
        bank_user = bank_users[bank_users.extype == e]
        bank_user = bank_user[bank_user.examount != 0]
        d[col + str(e) + '_min' + "_all"] = bank_user[col].min()
        d[col + str(e) + '_max' + "_all"] = bank_user[col].max()
        d[col + str(e) + '_median' + "_all"] = bank_user[col].median()
        d[col + str(e) + '_mean' + "_all"] = bank_user[col].mean()
        d[col + str(e) + '_std' + "_all"] = bank_user[col].std()
        d[col + str(e) + '_cnt' + "_all"] = bank_user[col].count()
        d[col + str(e) + '_max_min' + "_all"] = bank_user[col].max() - bank_user[col].min()

        d[col + str(e) + 'time_min'] = np.min(bank_users.time - bank_users.loan_time)
        d[col + str(e) + 'time_max'] = np.max(bank_users.time - bank_users.loan_time)

        di = bank_user.loc[bank_user.time >= bank_user.loan_time, :]
        d[col + str(e) + '_min' + "_gt"] = di[col].min()
        d[col + str(e) + '_max' + "_gt"] = di[col].max()
        d[col + str(e) + '_median' + "_gt"] = di[col].median()
        d[col + str(e) + '_mean' + "_gt"] = di[col].mean()
        d[col + str(e) + '_std' + "_gt"] = di[col].std()
        d[col + str(e) + '_cnt' + "_gt"] = di[col].count()
        d[col + str(e) + '_var' + "_gt"] = di[col].var()
        d[col + str(e) + '_max_min' + "_gt"] = di[col].max() - di[col].min()

        di = bank_user.loc[bank_user.time < bank_user.loan_time, :]
        d[col + str(e) + '_min' + "_lt"] = di[col].min()
        d[col + str(e) + '_max' + "_lt"] = di[col].max()
        d[col + str(e) + '_median' + "_lt"] = di[col].median()
        d[col + str(e) + '_mean' + "_lt"] = di[col].mean()
        d[col + str(e) + '_std' + "_lt"] = di[col].std()
        d[col + str(e) + '_cnt' + "_lt"] = di[col].count()
        d[col + str(e) + '_var' + "_lt"] = di[col].var()
        d[col + str(e) + '_max_min' + "_lt"] = di[col].max() - di[col].min()
    print d
    return d


stage = ['stg1_','stg2_','stg3_','stg4_','stg5_']
t = bank_data['time'].describe(percentiles=[0.2, 0.4, 0.6, 0.8])
split_point = [0, int(t['20%']), int(t['40%']), int(t['60%']), int(t['80%']), 1e11]

def bank_time_split_5(u):
    d = {'userid': u}
    for e in [0,1]:
        bank_user = bank_data[bank_data.userid == u]
        data = bank_user[bank_user.extype==e]
        for col in ['examount']:
            for i in range(5):
                    stg = stage[i]
                    di = data[(split_point[i]<(data.time))&((data.time)<split_point[i+1])]
                    d[stg+col+str(e)+'_min'] = di[col].min()
                    d[stg+col+str(e)+'_max'] = di[col].max()
                    d[stg+col+str(e)+'_median'] = di[col].median()
                    d[stg+col+str(e)+'_mean'] = di[col].mean()
                    d[stg+col+str(e)+'_std'] = di[col].std()
                    d[stg+col+str(e)+'_cnt'] = di[col].count()
                    d[stg + col + str(e) + '_var'] = di[col].var()
                    d[stg+col+str(e)+'_max_min'] = di[col].max() - di[col].min()
                    del di
            del data
    print d
    return d




def multi_bank():
    from multiprocessing import Pool
    pool = Pool(8)
    users = list(bank_data.userid.unique())
    for fun in [bank_time_split_2,bank_time_split_5]:

        rst = pool.map(fun, users)
        pool.close()
        pool.join()
        df = pd.DataFrame(rst)
        fileName = '../data/{}.csv'.format(fun.__name__)
        df.to_csv(fileName,index=None)


if __name__=='__main__':
    multi_bank()








