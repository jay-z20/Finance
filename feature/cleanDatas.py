# coding:utf-8


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt


def get_bill_data():

    names = ["userid", "time", "bank_id", "pre_amount_of_bill", "pre_repayment", "credit_amount", \
             "amount_of_bill_left", "least_repayment", "consume_amount", "amount_of_bill", "adjust_amount", \
             "circ_interest", "avail_amount", "prepare_amount", "repayment_state"]

    bill_train = pd.read_csv("../data/bill_detail_train.txt", header=None)
    bill_test = pd.read_csv("../data/bill_detail_test.txt", header=None)
    bill_data = pd.concat([bill_train, bill_test])
    bill_data.columns = names
    del bill_train, bill_test
    bill_data.loc[bill_data['consume_amount'] > 60, 'consume_amount'] = 60
    bill_data.loc[bill_data['prepare_amount'] > 25, 'prepare_amount'] = 23
    return bill_data

def bill_delete_duplicates(bill_data):

    bill_data.drop_duplicates(['userid','time','bank_id','pre_amount_of_bill','pre_repayment'],inplace=True)

    bill_data.to_csv('../data/bill_no_duplicate.csv',index=None)

bill_delete_duplicates(get_bill_data())


def get_bank_data():
    names = ['userid', 'time', 'extype', 'examount', 'mark']
    bank_detail_train = pd.read_csv("../data/bank_detail_train.txt", header=None)
    bank_detail_test = pd.read_csv("../data/bank_detail_test.txt", header=None)

    bank_detail = pd.concat([bank_detail_train, bank_detail_test])
    bank_detail.columns = names
    del bank_detail_train, bank_detail_test
    return bank_detail

def bank_delete_duplicates(bank_data):
    bank_data.drop_duplicates(['userid','time','extype','examount'],inplace=True)
    bank_data.to_csv('../data/bank_no_duplicate.csv',index=None)

bank_delete_duplicates(get_bank_data())

























