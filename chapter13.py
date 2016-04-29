# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
随机游动和数据可视化
"""
import random
import matplotlib.pylab as plt
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

class Location(object):
    """
    位置类：<x, y>
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self, deltaX, deltaY):
        """
        移动
        """
        return Location(self.x+deltaX, self.y+deltaY)
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def distFrom(self, other):
        xDist = self.x - other.x
        yDist = self.y - other.y
        return (xDist**2 + yDist**2)**0.5
    def __str__(self):
        return "<{0}, {1}>".format(self.x, self.y)

class Drunk(object):
    """
    醉汉类
    """
    def __init__(self, name=None):
        if name is not None and not isinstance(name, str):
            raise ValueError('Name should be a string type.')
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        return "Anonymous"

class UsualDunck(Drunk):
    """
    """
    def takeStep(self):
        stepChoices = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return random.choice(stepChoices)
class ColdDrunk(Drunk):
    def takeStep(self):
        stepChoices =[(0, 1), (0, -2), (1, 0), (-1, 0)]
        return random.choice(stepChoices)
class EWDrunk(Drunk):
    def takeStep(self):
        stepChoices = [(1, 0), (-1, 0)]
        return random.choice(stepChoices)

class Field(object):
    def __init__(self):
        self.drunks = {}

    def addDrunk(self, drunk, loc):
        if drunk in self.drunks:
            raise ValueError("Deplicate drunk")
        else:
            self.drunks[drunk] = loc

    def moveDrunk(self, drunk):
        if drunk not in self.drunks:
            raise ValueError("Drunk not in field")
        xDist, yDist = drunk.takeStep()
        currentLocation = self.drunks[drunk]
        self.drunks[drunk] = currentLocation.move(xDist, yDist)

    def getLoc(self, drunk):
        if drunk not in self.drunks:
            raise ValueError("Drunk not in field")
        return self.drunks[drunk]
class oddField(Field):
    def __init__(self, numHoles, xRange, yRange):
        Field.__init__(self)
        self.wormholes= {}
        for w in range(numHoles):
            x = random.randint(-xRange, xRange)
            y = random.randint(-yRange, yRange)
            newX = random.randint(-xRange, xRange)
            newY = random.randint(-yRange, yRange)
            newLoc = Location(newX, newY)
            self.wormholes[(x, y)] = newLoc
    def moveDrunk(self, drunk):
        Field.moveDrunk(self, drunk)
        x = self.drunks[drunk].getX()
        y = self.drunks[drunk].getY()
        if (x, y) in self.wormholes:
            self.drunks[drunk] = self.wormholes[(x, y)]

def walk(f, d, numSteps):
    """f是Field,d是drunk,numsteps是移动次数
    """
    start = f.getLoc(d)
    for s in range(numSteps):
        f.moveDrunk(d)
    return start.distFrom(f.getLoc(d))

def simWalks(numSteps, numTrials, dClass):
    """
    numSteps,移动步数
     numTrials, 实验次数
     dClass  Drunk的子类
    """
    Homer = dClass()
    origin = Location(0, 0)
    distances= []
    for t in range(numTrials):
        f= Field()
        f.addDrunk(Homer, origin)
        distances.append(walk(f, Homer, numSteps))
    return distances

def drunkTest(walkLengths, numTrials, dClass):
    means = []
    xVals = []
    stepRoots = []
    for numSteps in walkLengths:
        distances = simWalks(numSteps, numTrials, dClass)
        means.append(sum(distances)/len(distances))
        xVals.append(numSteps)
        stepRoots.append(numSteps**0.5)
        print(dClass.__name__, 'random walk of', numSteps, 'steps')
        print('Mean =', sum(distances)/len(distances), 'CV =', CV(distances))
        print("Max =", max(distances), 'Min =', min(distances))
    plt.plot(xVals, means, 'b.', label='Mean Distance')
    plt.plot(xVals, stepRoots, 'r-', label='The Root of Step')
    plt.legend(loc='best')
    plt.semilogx()
    plt.semilogy()
    plt.show()
class StyleIterator(object):
    """样式迭代器
    """
    def __init__(self, styles):
        self.index = 0
        self.styles = styles

    def nextStyle(self):
        result = self.styles[self.index]
        if self.index == len(self.styles)-1:
            self.index = 0
        else:
            self.index += 1
        return result
def simDrunk(numTrials, dClass, walkLengths):
    meanDistances = []
    cvDistances = []
    for numSteps in walkLengths:
        print("Starting simulation of", numSteps, 'steps')
        trials = simWalks(numSteps, numTrials, dClass)
        mean = sum(trials)/len(trials)
        meanDistances.append(mean)
        cvDistances.append(stdDev(trials)/mean)
    return meanDistances, cvDistances

def simAll(drunkKinds, walkLengths, numTrials):
    styleChoice = StyleIterator(('b-', 'r:', 'm-.'))
    for dClass in drunkKinds:
        curStyle = styleChoice.nextStyle()
        print("Starting simulation of", dClass.__name__)
        means, cvs = simDrunk(numTrials, dClass, walkLengths)
        cvMean = sum(cvs)/len(cvs)
        plt.plot(walkLengths, means, curStyle, label=dClass.__name__+'(CV ='+str(round(cvMean,4))+')')
    plt.title('Mean Distance from Origin (' + str(numTrials) +"trials")
    plt.xlabel('Number of Steps')
    plt.ylabel("Distance from Origin")
    plt.legend(loc='best')
    plt.semilogx()
    plt.semilogy()
    plt.savefig('MeanDistancefromOrigin.jpg', dpi=100)
    plt.show()

def getFinalLocs(numSteps, numTrials, dClass):
    locs = []
    d = dClass()
    origin = Location(0, 0)
    for t in range(numTrials):
        f = Field()
        f.addDrunk(d, origin)
        for s in range(numSteps):
            f.moveDrunk(d)
        locs.append(f.getLoc(d))
    return locs

def plotLocs(drunkKinds, numSteps, numTrials):
    styleChoice = StyleIterator(('b+', 'r^', 'mo'))
    for dClass in drunkKinds:
        locs = getFinalLocs(numSteps, numTrials, dClass)
        xVals, yVals = [], []
        for l in locs:
            xVals.append(l.getX())
            yVals.append(l.getY())
        meanX = sum(xVals)/len(xVals)
        meanY = sum(yVals)/len(yVals)
        curStyle = styleChoice.nextStyle()
        plt.plot(xVals, yVals, curStyle, label=dClass.__name__+" Mean loc. = <"+str(meanX)+', '+str(meanY)+'>')
    plt.title('Location at End of Walks (' + str(numSteps) + " steps)")
    plt.xlabel('Steps East/West of Origin')
    plt.ylabel('Steps North/South of Origin')
    plt.legend(loc='lower left', numpoints=1)
    plt.savefig('LocationatEndofWalks.jpg', dpi=100)
    plt.show()

def traceWalk(drunkKinds, numSteps):
    styleChoice = StyleIterator(('b+', 'r^', 'mo'))
    #f = Field()
    f = oddField(1000, 100, 200)
    for dClass in drunkKinds:
        d = dClass()
        f.addDrunk(d, Location(0, 0))
        locs = []
        for s in range(numSteps):
            f.moveDrunk(d)
            locs.append(f.getLoc(d))
        xVals, yVals = [], []
        for l in locs:
            xVals.append(l.getX())
            yVals.append(l.getY())
        curStyle = styleChoice.nextStyle()
        plt.plot(xVals, yVals, curStyle, label=dClass.__name__)
    plt.title('Spots Visited on Walk ('+str(numSteps)+' steps)')
    plt.xlabel('Steps East/West of Origin')
    plt.ylabel('Steps North/South of Origin')
    plt.legend(loc='best')
    plt.savefig('Spots2.jpg', dpi=100)
    plt.show()


if __name__ == "__main__":
    #drunkTest((10, 100, 1000, 10000, 100000 ), 100, UsualDunck)
    #simAll((UsualDunck, ColdDrunk, EWDrunk), (100, 1000, 10000, 100000), 100)
    #plotLocs((UsualDunck, ColdDrunk, EWDrunk), 100, 200)
    traceWalk((UsualDunck, ColdDrunk, EWDrunk), 200)
