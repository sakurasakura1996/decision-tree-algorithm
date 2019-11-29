import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
import pickle
import csv
import pandas as pd
#load data
# openfile = open("horse-colic.data",'r')
# data = pickle.load(openfile)
#
#
# train_data = []
# for line in data:
#     train_data.append(line)
# train_data=np.array(train_data)
# print(train_data.shape)
# print(train_data)

data_df = pd.read_csv('horse-colic.csv',header=None)   # 这里报错说每行有29个数据，原因是最后一列后面有一个空格导致的该问题
# 仔细排查了一下，发现，问题还不在这，后面我设置一个lineterminator的参数，就是每行以空格加换行结束，但是第一行默认是没有空格直接换行的，所以不能统一
# 暂时去除header=None属性，把第一行当作类名来算,妈的还是不对。
# 查询一下sep，delimiter，header等属性的含义
for line in data_df:
    curline =

print(data_df.shape)
#define and train the model
# clf = AdaBoost