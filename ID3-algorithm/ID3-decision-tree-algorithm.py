import numpy as np
import csv
import operator

def loadDataSet():
    dataSet = open('./isFish.csv','r')
    dataSet = csv.reader(dataSet)
    datalist =[]
    for line in dataSet:
        datalist.append(line)
    datalist = np.array(datalist)
    labelSet = datalist[:,-1]
    dataSet = np.array(datalist)
    labelSet = list(labelSet)
    # print(type(dataSet))
    # print(type(labelSet))
    # print(dataSet.shape)
    # print(labelSet.shape)
    print(type(labelSet))
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


def splitDataSet(dataSet,i,value):
    """
    这个函数可以说在编码时是比较难直接知道要定义一个实现这样的功能的函数的，后面需要用到的时候才会想到
    此函数实现，在寻找最佳划分属性时，每个属性下有不同取值，对应不同取值就会出现不同的子集和
    这里就是从dataSet中取出第i个特征 取值为value的子集和
    :param dataSet: 数据集
    :param i: 特征
    :param value: 特征的某个取值
    :return: subDataSet  第i个特征 取值为value的子集和
    """
    # 注意这里提取出来子集合是为了后面计算该子集合的信息熵所用，所以，目标应该是把第i个特征的取值为value的给去掉
    subDataSet  =[]
    for featVec in dataSet:
        # 将相同的特征取出来
        if featVec[i] == value:
            redeceDataSet = list(featVec[:i])   #注意转换成list后面要调用extend方法
            redeceDataSet.extend(featVec[i+1:])
            subDataSet.append(redeceDataSet)
    return subDataSet


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
        featureSet = set(featureList)
        miniEnt = 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = float(len(subDataSet))/float(len(dataSet))
            # num_mini_feature = featureList.count(value)
            # prob = float(num_mini_feature)/float(len(dataSet))
            # miniEnt -=prob * np.log2(prob)
            # 注意上面的计算是错误的，miniEnt计算时是比率乘以对应特征取值下的子集合的信息熵，正确写法如下
            miniEnt += prob * calShannonEnt(subDataSet)
            # 所以这里才想到上面还要加一个函数splitDataSet(dataSet,i,value)函数用于取到对应特征取值下的子集合
            # 不然完全没必要写什么splitDataSet函数，就如我上面的写法就可以了呀。因为上面写的不多，所以注释了
        inforGain = baseEnt - miniEnt
        if inforGain >bestInfoGain:
            bestInfoGain = inforGain
            bestFeature = i
    return bestFeature

# ok,现在已经能够选择出当前最佳划分的特征，接下来怎么做呢
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount =sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
# 这个函数也算是从后面代码中需要它时才会想到写这么一个函数


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 类别与属性完全相同时停止
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时，返回数量最多的
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)

    # 获取最佳划分属性
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 清空labels[beatFeat]
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归调用创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


dataSet,labelSet = loadDataSet()
shannonEnt = calShannonEnt(dataSet)
tree = createTree(dataSet,labelSet)
print('Decision Tree Algorithm ID3: \n {}'.format(tree))


