import numpy as np
import csv


def loadDataSet():
    dataSet = open('./isFish.csv','r')
    dataSet = csv.reader(dataSet)
    datalist =[]
    for line in dataSet:
        datalist.append(line)
    datalist = np.array(datalist)
    labelSet = datalist[:,-1]
    dataSet = np.array(datalist)
    labelSet = np.array(labelSet)
    # print(type(dataSet))
    # print(type(labelSet))
    # print(dataSet.shape)
    # print(labelSet.shape)
    return dataSet,labelSet


def calShannonEnt(dataSet):
    # 根据数据集来计算整个数据集D的信息熵（香农熵），联想公式进行编码
    # 首先，你最起码要知道最后的分类有多少种把，这里只有两种
    labelDict ={}
    for featVec in dataSet:
        if featVec[-1] not in labelDict.keys():
            labelDict[featVec[-1]]=0
        labelDict[featVec[-1]] +=1
    # 现在labelDict中存储着所有种类以及对应出现的次数，现在就准备计算信息熵了
    shannonEnt = 0.0
    numCount = len(dataSet)
    for key in labelDict:
        #计算 每类样本出现的比率
        prob = float(labelDict[key]/numCount)
        shannonEnt -=prob * np.log2(prob)
    return shannonEnt


def chooseBestFeature(dataSet):
    # 从数据集中找到当前最佳的划分属性，我们思考一下如何找到最好的划分属性呢
    # 因为ID3算法是根据每个特征的信息增益来划分的，所以要计算出每个特征的信息增益，将最大的作为划分属性
    numFeature = len(dataSet[0])-1
    baseEnt = calShannonEnt(dataSet)   # 总的信息熵，它减去对应特征的信息熵就是其信息增益
    bestInfoGain = 0.0 #初始化最大增益
    bestFeature = -1,  #初始化最优特征的索引
    for i in range(numFeature):
        # 循环体内计算第i个特征的信息增益，然后不断更新当前最大的信息增益和对应索引值
        # 这里就和上面计算总体的信息熵很相似了。首先你要知道每个特征有几种取值，然后算每种取值的概率，最后计算信息熵
        # 现在，第一步就是找到第i个特征的所有取值列表，
        featureList = [example[i] for example in dataSet]
