# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
蒙特卡罗模拟
"""
import random

def stdDev(X):
    """计算列表中元素方差
    """
    mean = sum(X)/len(X)
    std = 0
    for e in X:
        std += (e-mean)**2
    return (std/len(X))**0.5

def rollDie():
    return random.choice([1, 2, 3, 4, 5, 6])

def checkPascal(numTrials):
    """
    :param (int) numTrials：实验次数
    :return (float) :获胜的概率
    """
    numWins = 0
    for i in range(numTrials):
        for j in range(24):
            d1 = rollDie()
            d2 = rollDie()
            if d1 == 6 and d2 == 6:
                numWins += 1
                break
    return numWins/numTrials

class CrapsGame(object):
    """
    双骰子赌博游戏
    """
    def __init__(self):
        #记录出线与未过线的次数
        self.passWins, self.passLosses = 0, 0
        #记录赢、平局、输的次数
        self.dpWins, self.dpLosses, self.dpPushes = 0, 0, 0

    def palyHand(self):
        throw = rollDie() + rollDie()
        if throw == 7 or throw == 11:
            self.passWins += 1
            self.dpLosses += 1
        elif throw == 2 or throw == 3 or throw == 12:
            self.passLosses += 1
            if throw == 12:
                self.dpPushes += 1
            else:
                self.dpWins += 1
        else:
            point = throw
            while True:
                throw = rollDie() + rollDie()
                if throw == point:
                    self.passWins += 1
                    self.dpLosses += 1
                    break
                elif throw == 7:
                    self.passLosses += 1
                    self.dpWins += 1
                    break

    def fastPlayHand(self):
        """快速游戏
        """
        pointsDict = {4:1/3, 5:2/5, 6:5/11, 8:5/11, 9:2/5, 10:1/3}
        throw = rollDie() + rollDie()
        if throw == 7 or throw == 11:
            self.passWins += 1
            self.dpLosses += 1
        elif throw == 2 or throw == 3 or throw == 12:
            self.passLosses += 1
            if throw == 12:
                self.dpPushes += 1
            else:
                self.dpWins +=1
        else:
            if random.random() < pointsDict[throw]:
                self.passWins += 1
                self.dpLosses += 1
            else:
                self.passLosses += 1
                self.dpWins += 1

    def passResults(self):
        return self.passWins, self.passLosses

    def dpResults(self):
        return self.dpWins, self.dpPushes, self.dpLosses

def crapSim(handsPerGame, numGames):
    """
    玩每一局有handsPerGame次的游戏numGames次并输出结果
    """
    games = []
    #玩handsPerGames局游戏
    for t in range(numGames):
        c = CrapsGame()
        for i in range(handsPerGame):
            c.fastPlayHand()
            #c.palyHand()
        games.append(c)

    #生成每局游戏的统计结果
    pROIPerGame, dpROIPerGame = [], []
    for g in games:
        wins, losses = g.passResults()
        pROIPerGame.append((wins-losses)/handsPerGame)
        wins, pushes, losses = g.dpResults()
        dpROIPerGame.append((wins-losses)/handsPerGame)

    #生成并输出总结
    meanROI = str(round(100*sum(pROIPerGame)/numGames, 4)) + "%"
    sigma = str(round(100*stdDev(pROIPerGame), 4)) + "%"
    print("Pass:", "Mean POI =", meanROI, "Std. Dev =", sigma)
    meanROI = str(round(100 * sum(dpROIPerGame) / numGames, 4)) + "%"
    sigma = str(round(100 * stdDev(dpROIPerGame), 4)) + "%"
    print("Don't Pass:", "Mean POI =", meanROI, "Std. Dev =", sigma)

def throwNeedles(numNeedles):
    """
    模拟扔针过程
    """
    inCircle = 0
    for needles in range(numNeedles):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) <= 1.0:
            inCircle += 1
    return 4*(inCircle/numNeedles)

def getEst(numNeedles, numTrials):
    estimates = []
    for t in range(numTrials):
        piGuess = throwNeedles(numNeedles)
        estimates.append(piGuess)
    sDev = stdDev(estimates)
    curEst = sum(estimates)/len(estimates)
    print("Est. = " + str(round(curEst, 5)) +\
          ", Std. dev. = " + str(round(sDev, 5)) +\
          ", Needles = " + str(numNeedles))
    return curEst, sDev

def estPi(precision, numTrials):
    numNeedles = 1000
    sDev = precision
    while sDev >= precision/2.0:
        curEst, sDev = getEst(numNeedles, numTrials)
        numNeedles *= 2
    return curEst

if __name__ == "__main__":
    #print(checkPascal(100000))
    #crapSim(200000, 100)
    print(estPi(0.001, 100))