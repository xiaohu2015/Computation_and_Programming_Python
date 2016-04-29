# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
15：理解实验数据
"""
import matplotlib.pylab as plt
import numpy as np
import random
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['lines.linewidth'] = 2.5

def getData(fileName):
    dataFile = open(fileName, 'r')
    distances = []
    masses = []
    discardHeader = dataFile.readline()
    for line in dataFile:
        d, m = line.split(" ")
        distances.append(float(d))
        masses.append(float(m))
    dataFile.close()
    return masses, distances

def plotData(inputFile):
    masses, distances = getData(inputFile)
    masses = np.array(masses)
    distances = np.array(distances)
    forces = masses*9.81
    plt.plot(forces, distances, 'bo', label="Measured displacements")
    plt.title("测得的弹簧位移", fontproperties='SimHei')
    plt.xlabel('|阻力|(牛顿)', fontproperties='SimHei')
    plt.ylabel("距离 (米)", fontproperties='SimHei')
    plt.savefig('chapter15_1.jpg', dpi=100)
    plt.show()

def fitData(inputFile):
    masses, distances = getData(inputFile)
    #newMasses = masses + [x/100 for x in range(105, 155, 5)]
    masses = np.array(masses[:-6])
    distances = np.array(distances[:-6])
    forces = masses * 9.81
    #newForces = np.array(newMasses)*9.81
    plt.plot(forces, distances, 'bo', label="Measured displacements")
    plt.title("测得的弹簧位移")
    plt.xlabel('|阻力|(牛顿)')
    plt.ylabel("距离 (米)")
    #最小二乘法线性拟合(1次)
    a, b = plt.polyfit(forces, distances, 1)
    predictDistances = a*forces + b
    k = 1/a
    plt.plot(forces, predictDistances, label="预测线性拟合k="+str(round(k,5))+"的位移")

    '''
    # 最小二乘法线性拟合(3次)
    a, b, c, d = plt.polyfit(forces, distances, 3)
    predictDistances = a*(forces**3) + b*(forces**2) + c*forces + d
    plt.plot(forces, predictDistances, "r:", label="三次拟合")
    '''

    plt.legend(loc='best')
    plt.savefig('chapter15_5.jpg', dpi=100)
    plt.show()
    '''
    plt.figure(1)
    plt.plot(forces, distances, 'bo', label="Measured displacements")
    plt.title("测得的弹簧位移")
    plt.xlabel('|阻力|(牛顿)')
    plt.ylabel("距离 (米)")
    predictDistances = a * (newForces ** 3) + b * (newForces ** 2) + c * newForces + d
    plt.plot(newForces, predictDistances, "r:", label="三次拟合")
    plt.savefig('chapter15_4.jpg', dpi=100)
    plt.show()
    '''

def getTrajectoryData(fileName):
    dataFile = open(fileName, 'r')
    distances = []
    heights = []
    discardHeader = dataFile.readline()
    for line in dataFile:
        d, h1, h2, h3, h4 = list(map(float, line.split(" ")))
        distances.append(d)
        heights.append([h1, h2, h3, h4])
    dataFile.close()
    return distances, heights

def rSquared(measured, predicted):
    """
    :param (ndarray) measured  实测值
    :param (ndarray) predicted 预测值
    :return 决定系数
    """
    estimateError = np.sum(((predicted-measured)**2))
    meanOfMeasured = np.sum(measured)/measured.shape[0]
    variability = np.sum((measured-meanOfMeasured)**2)
    return 1 - estimateError/variability

def getHorizontalSpeed(a, b, c, minX, maxX):
    """
    :return 平均水平速度
    """
    inchesPerFoot = 12.0
    xMid = (maxX-minX)/2
    yPeak = a*(xMid**2) + b*xMid + c
    g = 32.16*inchesPerFoot
    t = (2*yPeak/g)**0.5
    print("HorizontalSpeed =", int(xMid/(t*inchesPerFoot)), "feet/sec")

def processTrajectories(fileName):
    distances, heights = getTrajectoryData(fileName)
    numTrials = len(heights[0])
    distances = np.array(distances)
    meanHeights = list(map(lambda x: sum(x)/len(x), heights))
    meanHeights = np.array(meanHeights)
    plt.title("抛弹的弹道("+str(numTrials)+"条弹道的平均路径)")
    plt.xlabel("距离抛射点的英寸数")
    plt.ylabel("抛射点上方的英寸数")
    plt.plot(distances, meanHeights, 'bo', label="实验点")
    a, b = plt.polyfit(distances, meanHeights, 1)
    altitudes = a*distances + b
    print("RSquare of linear fit =", rSquared(meanHeights, altitudes))
    plt.plot(distances, altitudes, 'b', label="线性拟合")
    a, b, c = plt.polyfit(distances, meanHeights, 2)
    altitudes = a*(distances**2) + b*distances + c
    getHorizontalSpeed(a, b, c, distances[-1], distances[0])
    print("RSquare of quadratic fit =", rSquared(meanHeights, altitudes))
    plt.plot(distances, altitudes, 'b:', label="二次拟合")
    plt.legend()
    plt.savefig("chapter15_6.jpg", dpi=100)
    plt.show()

#指数分布拟合
def f(x):
    '''定义一个指数函数
    '''
    return 3*(2**(1.2*x))

def createExpData(f, xVals):
    """
    :param f：指数函数

    """
    yVals = []
    for i in range(len(xVals)):
        yVals.append(f(xVals[i]))
    return np.array(xVals), np.array(yVals)

def fitExpData(xVals, yVals):
    logVals = []
    for y in yVals:
        logVals.append(math.log(y, 2.0))
    a, b = plt.polyfit(xVals, logVals, 1)
    return a, b, 2.0

def test():
    xVals, yVals = createExpData(f, list(range(10)))
    plt.plot(xVals, yVals, 'ro', label="实际数据")
    a, b, base = fitExpData(xVals, yVals)
    predictedYVals = []
    for x in xVals:
        predictedYVals.append(base**(a*x+b))
    plt.plot(xVals, predictedYVals, label="预测值")
    plt.title("拟合指数函数")
    plt.legend()
    plt.savefig("chapter16_7.jpg", dpi=100)
    plt.show()
    print("f(20) =", f(20))
    print("Predicted f(20) =", base**(a*20+b))

if __name__ == "__main__":
    #plotData("springData.txt")
    #fitData("springData.txt")
    #processTrajectories("trajectoryData.txt")
    test()