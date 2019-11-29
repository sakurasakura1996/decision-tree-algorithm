import numpy as np

def loadSimpData():
    dataMat = np.array([[1.0,2.1],
                        [2.0,1.1],
                        [1.3,1.0],
                        [1.0,1.0],
                        [2.0,1.0]])
    classLabel = [1.0,1.0,-1.0,-1.0,1.0]
    return np.array(dataMat),np.array(classLabel)


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    简单的通过阈值比较进行数据分类
    :param dataMatrix: 训练样本数据
    :param dimen: 数据的第几列，也就是第几个特征
    :param threshVal: 阈值，如果大于它归为一类，小于它归为另一类
    :param threshIneq: lt 或者 gt 也就是决定大于或者小于
    :return: 分类情况
    """
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal] = -1.0
    return retArray


def buildStump(dataArr,classLabels,D):
    """
    构建一个决策树桩
    :param dataArr:训练数据集
    :param classLabels: 数据分类向量
    :param D: 权重向量
    :return: 当前构建的最小错误率的决策树
    """
    m, n = dataArr.shape[0],dataArr.shape[1]
    minError = np.inf
    bestStump = {}
    bestClassEst = np.zeros((m,1))
    numsteps = 10.0
    for i in range(n):
        rangeMin = dataArr[:,i].min()
        rangeMax = dataArr[:,i].max()
        stepSize = (rangeMax - rangeMin)/numsteps
        for j in (-1,int(numsteps+1)):
            for inequal in ['lt','gt']:
                threshVal = rangeMin+float(j)*stepSize
                predictedVals = stumpClassify(dataArr,i,threshVal,inequal)
                errArr = np.ones((m,1))
                errArr[predictedVals == classLabels] = 0
                weightedError = D.T*errArr
                print("split: dim %d,thresh %.2f,thresh inequal: %s, the weighted error is %.3f" %(i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst


def adaBoostTrain(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = dataArr.shape[0]
    D = (np.ones((m,1))/m)   #init D to all equal
    aggClassEst = np.zeros((m,1))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        print("error",error)
        alpha = float(0.5* np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        print("alpha",alpha)
        weakClassArr.append(bestStump)#store Stump Params in Array
        print("classEst",classEst)
        expon = np.multiply(-1*alpha*np.array((classLabels).T),classEst) #exponent for D calc, getting messy
        D = np.multiply(D,np.exp(expon)) #Calc New D, element-wise
        D = D/D.sum()
        print("D",D)
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print("aggClassEst",aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.array((classLabels).T),np.ones((m,1)))
        #print aggErrors
        errorRate = aggErrors.sum()/m
        print(errorRate)
        if errorRate == 0.0: break
    return weakClassArr


datamat, classlabel = loadSimpData()
classifierarray = adaBoostTrain(datamat,classlabel,9)


