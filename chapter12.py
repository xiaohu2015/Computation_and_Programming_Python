# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
统计推断与模拟
"""
import random
import matplotlib.pylab as plt
import numpy as np

def stdDev(X):
    """计算列表中元素方差
    """
    mean = sum(X)/len(X)
    std = 0
    for e in X:
        std += (e-mean)**2
    return (std/len(X))**0.5

def CV(X):
    mean = sum(X)/len(X)
    try:
        return stdDev(X)/mean
    except ZeroDivisionError:
        return float('nan')

def fip(numFlips):
    heads = 0
    for i in range(numFlips):
        if random.random() < 0.5:
            heads += 1
    return heads/numFlips

def flipSim(numFlipsPerTrial, numTrials):
    fracHeads = []
    for i in range(numTrials):
        fracHeads.append(fip(numFlipsPerTrial))
    mean = sum(fracHeads)/len(fracHeads)
    sd = stdDev(fracHeads)
    return fracHeads, mean, sd



def flipPlot(minExp, maxExp):
    """假定minEXPy和maxExp是正整数且minExp<maxExp
    绘制出2**minExp到2**maxExp次抛硬币的结果
    """
    ratios = []
    diffs = []
    aAxis = []
    for i in range(minExp, maxExp+1):
        aAxis.append(2**i)
    for numFlips in aAxis:
        numHeads = 0
        for n in range(numFlips):
            if random.random() < 0.5:
                numHeads += 1
        numTails = numFlips - numHeads
        ratios.append(numHeads/numFlips)
        diffs.append(abs(numHeads-numTails))
    plt.figure()
    ax1 = plt.subplot(121)
    plt.title("Difference Between Heads and Tails")
    plt.xlabel('Number of Flips')
    plt.ylabel('Abs(#Heads - #Tails)')
    ax1.semilogx(aAxis, diffs, 'bo')
    ax2 = plt.subplot(122)
    plt.title("Heads/Tails Ratios")
    plt.xlabel('Number of Flips')
    plt.ylabel("#Heads/#Tails")
    ax2.semilogx(aAxis, ratios, 'bo')
    plt.show()

def makePlot(xVals, yVals, title, xLabel, yLabel, style, logX=False, logY=False):
    """用给定的标题和标签绘制xVals和yVals
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(xVals, yVals, style)
    if logX:
        plt.semilogx()
    if logY:
        plt.semilogy()
def runTrail(numFlips):
    numHeads = 0
    for i in range(numFlips):
        if random.random() < 0.5:
            numHeads += 1
    numTails = numFlips - numHeads
    return numHeads, numTails



def flipPlot1(minExp, maxExp, numTrials):
    """假定minEXPy和maxExp是正整数且minExp<maxExp
        绘制出2**minExp到2**maxExp次抛硬币的结果
    """
    ratiosMeans, diffsMeans, ratiosSDs, diffsSDs = [], [], [], []
    ratiosCVs, diffsCVs = [], []
    xAxis = []
    for exp in range(minExp, maxExp):
        xAxis.append(2**exp)
    for numFlips in xAxis:
        ratios = []
        diffs = []
        for t in range(numTrials):
            numHeads, numTails = runTrail(numFlips)
            ratios.append(numHeads/numFlips)
            diffs.append(abs(numHeads-numTails))
        ratiosMeans.append(sum(ratios)/len(ratios))
        ratiosSDs.append(stdDev(ratios))
        diffsMeans.append(sum(diffs)/len(diffs))
        diffsSDs.append(stdDev(diffs))
        ratiosCVs.append(CV(ratios))
        diffsCVs.append(CV(diffs))
    numTrialsStr = ' (' + str(numTrials) + " Trials)"
    title = "Mean Heads/Tails Ratios" + numTrialsStr
    makePlot(xAxis, ratiosMeans, title, 'Number of flips', 'Mean Heads/Tails',
             'bo', logX=True)
    title = "SD Heads/Tails Ratios" + numTrialsStr
    makePlot(xAxis, ratiosSDs, title, 'Number of flips', 'Standard Deviation',
             'bo', logX=True, logY=True)
    title = "Mean Abs(#Heads - #Tails)" + numTrialsStr
    makePlot(xAxis, diffsMeans, title, 'Number of flips', 'Mean Abs(#Heads - #Tails)',
             'bo', logX=True, logY=True)
    title = "SD Heads/Tails Ratios" + numTrialsStr
    makePlot(xAxis, diffsSDs, title, 'Number of flips', 'Standard Deviation',
             'bo', logX=True, logY=True)
    title = "Coeff. of Var.  Abs(#Heads - #Tails)" + numTrialsStr
    makePlot(xAxis, diffsCVs, title, 'Number of flips', 'Coeff. of Var.',
             'bo', logX=True)
    title = "Coeff. of Var. Heads/Tails Ratios" + numTrialsStr
    makePlot(xAxis, ratiosCVs, title, 'Number of flips', 'Coeff. of Var.',
             'bo', logX=True, logY=True)

def labelPlot(numFlips, numTrials, mean, sd):
    plt.title(str(numTrials) + " trials of " + str(numFlips) + " flips each")
    plt.xlabel("Fraction of Heads")
    plt.ylabel("Number of Trials")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.text(xmin + (xmax-xmin)*0.02, (ymax-ymin)/2, 'Mean = ' + str(round(mean, 4))
             + '\nSD' + str(round(sd, 4)), size='x-large')
def makePlots(numFlips1, numFlips2, numTrials):
    val1, mean1, sd1 = flipSim(numFlips1, numTrials)
    plt.hist(val1, bins=20)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    labelPlot(numFlips1, numTrials, mean1, sd1)
    plt.figure()
    val2, mean2, sd2 = flipSim(numFlips2, numTrials)
    plt.hist(val2, bins=20)
    plt.xlim(xmin, xmax)
    labelPlot(numFlips2, numTrials, mean2, sd2)

def showErrorBars(minExp, maxRxp, numTrials):
    """绘制误差图
    """
    means, sds = [], []
    xVals = []
    for exp in range(minExp, maxRxp+1):
        xVals.append(2**exp)
    for numFlips in xVals:
        fracHeads, mean, sd = flipSim(numFlips, numTrials)
        means.append(mean)
        sds.append(sd)
    plt.errorbar(xVals, means, yerr=2*np.array(sds))
    plt.semilogx()
    plt.title('Mean Fraction of Heads (' + str(numTrials) + ' Trials)')
    plt.xlabel('Number of flips per trial')
    plt.ylabel("Fraction of heads & 95% confidence")

def clear(n, p, steps):
    """假定n和steps是正整数,p是浮点数,
    n是分子的初始数量
    p是分子被清除的概率
    steps是模拟的时间
    """
    numRemaining = [n]
    for t in range(1, steps+1):
        numRemaining.append(numRemaining[-1]*(1-p))
    plt.plot(numRemaining)
    plt.xlabel('Time')
    plt.ylabel('Molecules Remianing')
    plt.title('Clearance of Drug')
    plt.semilogy()

def successfulStarts(eventProb, numTrials):
    """eventProb是事件概率,numTrials是实验总次数
    返回每次实验成功需要尝试的次数列表
    """
    triesBeforeSucess = []
    for t in range(numTrials):
        consecFailures = 0
        while random.random() > eventProb:
            consecFailures += 1
        triesBeforeSucess.append(consecFailures)
    return triesBeforeSucess

def playSeries(numGames, teamProb):
    """假定numGames是奇数
    teamProb是获胜概率
    如果获胜返回true
    """
    numWon = 0
    for game in range(numGames):
        if random.random() <= teamProb:
            numWon += 1
    return numWon>numGames//2

def simSeries(numSeries):
    prob = 0.5
    fracWon = []
    probs = []
    while prob <= 1.0:
        seriesWon = 0
        for i in range(numSeries):
            if playSeries(7, prob):
                seriesWon += 1
        fracWon.append(seriesWon/numSeries)
        probs.append(prob)
        prob += 0.01
    plt.plot(probs, fracWon, linewidth=3)
    plt.xlabel("Probability of Winning a Game")
    plt.ylabel("Probability of Winning a Series")
    plt.axhline(0.95)
    plt.ylim(0.5, 1.1)
    plt.title(str(numSeries) + " Seven-Game Series")

def findSeriesLength(teamProb):
    numSeries = 200
    maxLen = 2500
    step = 10
    def fracWon(teamProb, numSeries, seriesLen):
        won = 0
        for series in range(numSeries):
            if playSeries(seriesLen, teamProb):
                won += 1
        return won/numSeries
    winFrac = []
    xVals = []
    for seriesLen in range(1, maxLen, step):
        xVals.append(seriesLen)
        winFrac.append(fracWon(teamProb, numSeries, seriesLen))
    plt.plot(xVals, winFrac, linewidth=3)
    plt.xlabel("Length of Series")
    plt.ylabel("Probability of Winning Series")
    plt.title(str(round(teamProb, 4))+ " Probability of Better Team Winning a Game")
    plt.axhline(0.95)
    plt.ylim(0.5, 1.1)

def simInsertions(numIndices, numInsertions):
    """发生碰撞返回1，否则返回0
    """
    choices = range(numIndices)
    used = []
    for i in range(numInsertions):
        hashVal = random.choice(choices)
        if hashVal in used:
            return 1
        else:
            used.append(hashVal)
    return 0

def findProb(numIndices, numInsertions, numTrials):
    collisions = 0
    for t in range(numTrials):
        collisions += simInsertions(numIndices, numInsertions)
    return collisions/numTrials

def collisionProb(n, k):
    prob = 1.0
    for i in range(1, k):
        prob = prob*((n-i)/n)
    return 1-prob

if __name__ == "__main__":
    print(flipSim(100, 10))
    random.seed(0)
    '''
    vals = [1, 200]
    for i in range(1000):
        num1 = random.choice(range(1, 100))
        num2 = random.choice(range(1, 100))
        vals.append(num1 + num2)
    plt.hist(vals, bins=10)
    '''
    #makePlots(100, 1000, 100000)
    #flipPlot1(4, 20, 20)
    #showErrorBars(3, 10, 100)
    #clear(1000, 0.01, 1000)
    '''
    probOfSuccess = 0.5
    numTrials = 5000
    distribution = successfulStarts(probOfSuccess, numTrials)
    plt.hist(distribution, bins=14)
    plt.title("Probability of Starting Each Try "+ str(probOfSuccess))
    plt.xlabel("Tries Before Success")
    plt.ylabel("Number of Occurrences Out of " + str(numTrials))
    '''
    #simSeries(400)
    '''
    YanksProb = 0.636
    PhilsProb = 0.574
    findSeriesLength(YanksProb/(YanksProb+PhilsProb))
    '''

    print(collisionProb(1000, 50))
    print(findProb(1000, 50, 10000))
    plt.show()