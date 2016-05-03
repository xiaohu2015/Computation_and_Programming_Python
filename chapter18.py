# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
第18章：动态规划
"""
import random

numCalls = 0

def fastFib(n, memo={}):
    """
    快速斐波那契数列实现
    :param (int) n: 要求解的项数
    :param memo: 记录已经计算的项
    :return : n的斐波那契数
    """
    if n == 0 or n == 1:
        return 1
    try:
        return memo[n]
    except KeyError:
        result = fastFib(n-1, memo) + fastFib(n-2, memo)
        memo[n] = result
        return result

#动态规划，0/1背包问题

class Item(object):
    def __init__(self, n, v, w):
        self.__name = n
        self.__value = v
        self.__weight = w
    def getName(self):
        return self.__name
    def getValue(self):
        return self.__value
    def getWeight(self):
        return self.__weight
    def __str__(self):
        result = "<" + self.__name + ", " + str(self.__value)\
                + ", " + str(self.__weight) + ">"
        return result

def printResult(result):
    '''
    输出结果
    '''
    print("Total value of items taken:", result[0], end='   ')
    for item in result[1]:
        print(item, end=' ')
    print()

def maxVal(toConsider, avail):
    '''
    求解最优背包
    :param toConsider:目前背包能选择的物品
    :param avail:目前背包的剩余容量
    :return ：返回最优背包列表及最大价值
    '''
    if len(toConsider) == 0 or avail == 0:
        return (0, ())
    elif toConsider[0].getWeight() > avail:
        #此时只遍历右分枝
        return maxVal(toConsider[1:], avail)
    else:
        nextItem = toConsider[0]
        #遍历左分支
        withVal, withToTake = maxVal(toConsider[1:], avail-nextItem.getWeight())
        withVal += nextItem.getValue()
        #遍历右分枝
        withoutVal, withoutToTake = maxVal(toConsider[1:], avail)
        #选择更优的分支
        if withVal > withoutVal:
            result = (withVal, withToTake+(nextItem,))
        else:
            result = (withoutVal, withoutToTake)
    #printResult(result)
    return result

def fastMaxVal(toConsider, avail, memo={}):
    """
    快速背包算法，memo在递归中调用，记录已经计算的重复子问题
    """
    if (len(toConsider), avail) in memo:
        return memo[(len(toConsider), avail)]
    elif toConsider == [] or avail == 0:
        result = (0, ())
    elif toConsider[0].getWeight() > avail:
        result = fastMaxVal(toConsider[1:], avail, memo)
    else:
        nextItem = toConsider[0]
        # 遍历左分支
        withVal, withToTake = fastMaxVal(toConsider[1:], avail - nextItem.getWeight(), memo)
        withVal += nextItem.getValue()
        # 遍历右分枝
        withoutVal, withoutToTake = fastMaxVal(toConsider[1:], avail, memo)
        # 选择更优的分支
        if withVal > withoutVal:
            result = (withVal, withToTake + (nextItem,))
        else:
            result = (withoutVal, withoutToTake)
    memo[(len(toConsider), avail)] = result
    global numCalls
    numCalls += 1
    return result

def smallTest():
    '''
    测试背包算法
    '''
    names = ['a', 'b', 'c', 'd']
    vals = [6, 7, 8, 9]
    weights = [3, 3, 2, 5]
    items = []
    for i in range(len(names)):
        items.append(Item(names[i], vals[i], weights[i]))
    result = maxVal(items, 5)
    printResult(result)

def buildManyItems(numItems, maxVal, maxWeight):
    """
    构建测试物品
    """
    items = []
    for i in range(numItems):
        items.append(Item(str(i), random.randint(1, maxVal),\
                          random.randint(1, maxWeight)))
    return items

def bigTest(numItems):
    """
    大测试
    """
    items = buildManyItems(numItems, 10, 10)
    '''
    val, taken = maxVal(items, 25)
    print("Items Taken:")
    for item in taken:
        print(item)
    print('Total value of items taken =', val)
    '''
    print("***********************************")
    val, taken = fastMaxVal(items, 20)
    print("Items Taken:")
    for item in taken:
        print(item)
    print('Total value of items taken =', val)
    print(numCalls)

if __name__ == "__main__":
    #print(fastFib(500))
    #smallTest()
    bigTest(100)