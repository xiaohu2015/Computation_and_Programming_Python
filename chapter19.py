# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
第19章：机器学习简介
"""
import numpy as np
import matplotlib.pylab as plt
import random

def stdDev(X):
    """计算列表中元素方差
    """
    mean = sum(X)/len(X)
    std = 0
    for e in X:
        std += (e-mean)**2
    return (std/len(X))**0.5

#闵科夫斯基距离计算
def minkowskiDist(v1, v2, p):
    """
    闵科夫斯基距离计算
    :param v1:特征向量
    :param v2:特征向量，v1与v2等长
    :param p:指数
    """
    #dist = 0
    dist = np.sum((np.abs(v1-v2))**p)
    '''
    for i in range(len(v1)):
        dist += abs(v1[i]-v2[i])**p
    '''
    return dist**(1/p)

#动物类
class Animal(object):
    '''
    动物类
    '''
    def __init__(self, name, features):
        '''
        :param (string) name:姓名
        :param (ndarray) features:特征向量
        '''
        self._name = name
        self._features = np.array(features)

    def getName(self):
        return self._name

    def getFeatures(self):
        return np.array(self._features)

    def distance(self, other):
        '''
        与其他动物之间距离
        欧式距离
        '''
        return minkowskiDist(self._features, other.getFeatures(), 2)

def compareAnimals(animals, precision):
    '''
    :param animals:动物列表
    :param (int) precision: 精度
    :return : 表格，包含任意两个动物之间的欧式距离
    '''
    columnLabels = []
    for a in animals:
        columnLabels.append(a.getName())
    rowLabels = columnLabels[:]
    tableVals = []
    #循环计算任意两个动物间的欧氏距离
    for a1 in animals:
        row =[]
        for a2 in animals:
            if a1 == a2:
                row.append('--')
            else:
                distance = a1.distance(a2)
                row.append(str(round(distance, precision)))
        tableVals.append(row)
    #生成表格
    table = plt.table(rowLabels=rowLabels,
                      colLabels=columnLabels,
                      cellText=tableVals,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2]*len(animals))
    table.scale(1, 2.5)
    plt.axis('off')
    plt.savefig('chapter19_1.png', dpi=100)
    plt.show()

def testDist():
    rattlesnake = Animal('rattlesnake', [1, 1, 1, 1, 0])
    boa = Animal('boa\nconstrictor', [0, 1, 0, 1, 0])
    dartFrog = Animal('dart frog', [1, 0, 1, 0, 4])
    alligator = Animal('alligator', [1, 1, 0, 1, 4])
    animals = [rattlesnake, boa, dartFrog, alligator]
    compareAnimals(animals, 3)

#聚类分析
class Example(object):
    '''
    聚类实例
    '''
    def __init__(self, name, features, label=None):
        '''
        :param (string) name:姓名
        :param (ndarray) features:特征向量,数字列表
        :param label: 标签
        '''
        self._name = name
        self._features = features
        self._label = label

    def dimensionality(self):
        '''
        获得特征向量的维度
        '''
        return len(self._features)

    def getFeatures(self):
        return np.array(self._features)

    def getLabel(self):
        return self._label

    def getName(self):
        return self._name

    def distance(self, other):
        return minkowskiDist(self.getFeatures(), other.getFeatures(), 2)

    def __str__(self):
        return self._name + ":" + str(self._features) + ":" +\
                str(self._label)

class Cluster(object):
    '''
    簇类
    '''
    def __init__(self, examples, exampleType):
        self.examples = examples
        self.exampleType = exampleType
        self.centroid = self.computerCentroid()

    def update(self, examples):
        '''
        用新实例替换簇中原有实例
        返回质心的改变量
        '''
        oldCentroid = self.centroid
        self.examples = examples
        if len(examples) > 0:
            self.centroid = self.computerCentroid()
            return oldCentroid.distance(self.centroid)
        else:
            return 0.0

    def members(self):
        for e in self.examples:
            yield e

    def size(self):
        return len(self.examples)

    def getCentroid(self):
        return self.centroid

    def computerCentroid(self):
        '''
        计算簇的质心
        '''
        dim = self.examples[0].dimensionality()
        totVals = np.array([0]*dim)
        for e in self.examples:
            totVals += e.getFeatures()
        centroid = self.exampleType('centroid', totVals/len(self.examples))
        return centroid

    def variance(self):
        '''
        计算方差
        '''
        totDist = 0.0
        for e in self.examples:
            totDist += (e.distance(self.centroid))**2
        return totDist**0.5

    def __str__(self):
        names = []
        for e in self.examples:
            names.append(e.getName())
        names.sort()
        result = 'Cluster with centroid ' + str(self.centroid.getFeatures()) + \
                    ' contains:\n'
        for e in names:
            result += e + ", "
        return result[:-2]
#kmeans聚类算法
def kmeans(examples, exampleType, k, verbose):
    '''
    假设examples是exampleType类型的实例列表
    k是整数，代表簇的数量
    verbose：布尔值，当为true时，输出每次迭代的结果
    '''
    #随机生成k个初始质心
    initialCentroids = random.sample(examples, k)
    #为每个质心创建一个单例簇
    clusters = []
    for e in initialCentroids:
        clusters.append(Cluster([e], exampleType))

    #不断迭代，直到质心不变
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        #创建一个列表，包含k个空列表
        newClusters = []
        for i in range(k):
            newClusters.append([])
        #每个实例关联到最近的质心
        for e in examples:
            #找到离e最近的质心
            samllestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1,k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < samllestDistance:
                    samllestDistance = distance
                    index = i
            #将e添加到对应簇的实例列表中
            newClusters[index].append(e)
        #更新所有簇，判断质心是否改变
        converged = True
        for i in range(len(clusters)):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False

        if verbose:
            print("Iteration #"+str(numIterations))
            for c in clusters:
                print(c)
            print('*****************************')

    return clusters

def dissimilarity(clusters):
    '''
    计算分类后的总方差
    '''
    totDist = 0
    for c in clusters:
        totDist += c.variance()
    return totDist

def tryKmeans(examples, exampleType, numClusters, numTrials, verbose=False):
    '''
    调用kmeans numTrials次并返回差异和最小的结果
    '''
    best = kmeans(examples, exampleType, numClusters, verbose)
    minDissimilarity = dissimilarity(best)
    for trial in range(1, numTrials):
        clusters = kmeans(examples, exampleType, numClusters, verbose)
        currDissimilarity = dissimilarity(clusters)
        if currDissimilarity < minDissimilarity:
            best = clusters
            minDissimilarity = currDissimilarity
    return best

def genDistribution(xMean, xSD, yMean, ySD, n, namePrefix):
    '''
    创建测试实例
    '''
    samples = []
    for s in range(n):
        x = random.gauss(xMean, xSD)
        y = random.gauss(yMean, ySD)
        samples.append(Example(namePrefix+str(s), [x, y]))
    return samples

def plotSamples(samples, marker):
    '''
    绘制样本
    '''
    xVals = []
    yVals = []
    for s in samples:
        x = s.getFeatures()[0]
        y = s.getFeatures()[1]
        plt.annotate(s.getName(), xy=(x,y), xytext=(x+0.13, y-0.07), fontsize='x-large')
        xVals.append(x)
        yVals.append(y)
    plt.plot(xVals, yVals, marker)

def contrivedTest(numTrials, k, verbose):
    '''
    测试
    '''
    random.seed(0)
    xMean = 3
    xSD = 1
    yMean = 5
    ySD = 1
    n = 10
    d1Samples = genDistribution(xMean, xSD, yMean, ySD, n, '1.')
    plotSamples(d1Samples, 'b^')
    d2Samples = genDistribution(xMean+3, xSD, yMean+1, ySD, n, '2.')
    plotSamples(d2Samples, 'ro')
    clusters = tryKmeans(d1Samples+d2Samples, Example, k, numTrials, verbose)
    print("Final result:")
    for c in clusters:
        print('   ', c)
    plt.savefig('chapter19_2.png',dpi=100)
    plt.show()

def contrivedTest2(numTrials, k, verbose):
    '''
    测试2
    '''
    random.seed(0)
    xMean = 3
    xSD = 1
    yMean = 5
    ySD = 1
    n = 8
    d1Samples = genDistribution(xMean, xSD, yMean, ySD, n, '1.')
    plotSamples(d1Samples, 'b^')
    d2Samples = genDistribution(xMean + 3, xSD, yMean, ySD, n, '2.')
    plotSamples(d2Samples, 'ro')
    d3Samples = genDistribution(xMean, xSD, yMean+3, ySD, n, '3.')
    plotSamples(d3Samples, 'gd')
    clusters = tryKmeans(d1Samples + d2Samples + d3Samples, Example, k, numTrials, verbose)
    print("Final result:")
    for c in clusters:
        print('   ', c)
    plt.savefig('chapter19_3.png', dpi=100)
    plt.show()

def scaleFeatures(vals):
    '''
    标准化属性
    '''
    mean = sum(vals)/len(vals)
    result = np.array(vals) - mean
    sd = stdDev(vals)
    result = result/sd
    return list(result)

#聚类实例：哺乳动物
def readMammalData(fName, scale):
    '''
    读取数据
    '''
    dataFile = open(fName, 'r')
    numFeatures = 0
    #处理文件头部

    for line in dataFile:   #计算特征数目
        if line[0:6] == '#Label':
            break
        if line[0:5] != "#Name":
            numFeatures += 1
    #
    featureVals, speciesNames, labelList = [], [], []
    for i in range(numFeatures):
        featureVals.append([])

    #处理特征向量
    for line in dataFile:
        dataLine = line[:-1].split(',')  #删除最后的换行符
        speciesNames.append(dataLine[0])
        classLabel = float(dataLine[-1])
        labelList.append(classLabel)
        for i in range(numFeatures):
            featureVals[i].append(float(dataLine[i+1]))
    if scale:
        for i in range(numFeatures):
            featureVals[i] = scaleFeatures(featureVals[i])
    dataFile.close()

    #生成特征向量
    featureVectorList = []
    for mammal in range(len(speciesNames)):
        featureVector = []
        for feature in range(numFeatures):
            featureVector.append(featureVals[feature][mammal])
        featureVectorList.append(featureVector)


    return featureVectorList, labelList, speciesNames

def bulidMammalExamples(featureList, labelList, speciesNames):
    '''
    创建实例
    '''
    examples = []
    for i in range(len(speciesNames)):
        example = Example(speciesNames[i], featureList[i], labelList[i])
        examples.append(example)
    return examples

def testTeeth(numClusters, numTrials, scale):
    features, labels, species = readMammalData('dentalFormulas.txt', scale)
    examples = bulidMammalExamples(features, labels, species)
    bestClustering = tryKmeans(examples, Example, numClusters, numTrials)
    for c in bestClustering:
        names = ''
        herbivores, carnivores, omnivores = 0, 0, 0
        for p in c.members():
            names += p.getName() + ', '
            if p.getLabel() == 0:
                herbivores += 1
            elif p.getLabel() == 1:
                carnivores += 1
            else:
                omnivores += 1
        print('\n', names[:-2])
        print(herbivores, 'herbivores', carnivores, 'carnivores', omnivores, 'omnivores')


if __name__ == "__main__":
    #testDist()
    #contrivedTest(1, 2, True)
    #contrivedTest2(40, 5, False)
    #f = open('dentalFormulas.txt', 'r')
    testTeeth(3, 20, False)
    print('')
    testTeeth(3, 20, True)
