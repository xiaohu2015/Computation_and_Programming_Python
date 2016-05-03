# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
import requests
"""
背包和图的最优化问题
"""
#背包问题
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

def value(item):
    return item.getValue()

def weightInverse(item):
    return 1.0/item.getWeight()

def density(item):
    return item.getValue()/item.getWeight()

def buildItems():
    names = ['clock', 'painting', 'radio', 'vase', 'book', 'computer']
    vals = [175, 90, 20, 50, 10, 200]
    weights = [10, 9, 4, 2, 1, 20]
    items = []
    for i in range(len(names)):
        items.append(Item(names[i], vals[i], weights[i]))
    return items

def greedy(items, maxWeight, keyFunction):
    sortedItems = sorted(items, key=keyFunction, reverse=True)
    result = []
    totalValue = 0
    totalWeight = 0
    for item in sortedItems:
        if (totalWeight + item.getWeight()) <= maxWeight:
            result.append(item)
            totalWeight += item.getWeight()
            totalValue += item.getValue()
    return result, totalValue

def testGreedy(items, constraint, keyFunction):
    taken, val = greedy(items, constraint, keyFunction)
    print("Total value of items taken =", val)
    for item in taken:
        print("   ", item)

def testGreedys(maxWeight=20):
    items = buildItems()
    print("Use greedy by value to fill knapsack of size", maxWeight)
    testGreedy(items, maxWeight, value)
    print("Use greedy by weight to fill knapsack of size", maxWeight)
    testGreedy(items, maxWeight, weightInverse)
    print("Use greedy by density to fill knapsack of size", maxWeight)
    testGreedy(items, maxWeight, density)
def getBinaryRep(n, numDigits):
    '''
    返回一个numDigits位字符串，内容是n的二进制表示
    '''
    result = ''

    while n > 0:
        result = str(n%2) + result
        n = n//2
    if len(result) > numDigits:
        raise ValueError("not enough digits.")
    for i in range(numDigits-len(result)):
        result = "0" + result
    return result

def genPowerset(L):
    '''
    返回集合中所有的子集
    '''
    powerset = []
    for i in range(0, 2**len(L)):
        binStr = getBinaryRep(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == "1":
                subset.append(L[j])
        powerset.append(subset)
    return powerset

def chooseBest(pset, maxWeight):
    bestVal = 0
    bestSet = None
    for items in pset:
        itemsVal = 0
        itemsWeight = 0
        for item in items:
            itemsVal += item.getValue()
            itemsWeight += item.getWeight()
            if itemsWeight > maxWeight:
                break
        if itemsWeight <= maxWeight and itemsVal > bestVal:
            bestVal = itemsVal
            bestSet = items
    return bestSet, bestVal

def testBest(maxWeight=20):
    items = buildItems()
    pset = genPowerset(items)
    taken, val = chooseBest(pset, maxWeight)
    print("Teotal value of items taken =", val)
    for item in taken:
        print(item)

#图问题
class Node(object):
    '''
    节点
    '''
    def __init__(self, name):
        '''
        :param string name:节点名称
        '''
        self._name = name

    def getName(self):
        return self._name
    def __str__(self):
        return self._name

class Edge(object):
    '''
    边
    '''
    def __init__(self, src, dest):
        '''
        :param Node src:源节点
        :param Node dest:终节点
        '''
        self._src = src
        self._dest = dest

    def getSource(self):
        return self._src
    def getDestination(self):
        return self._dest
    def __str__(self):
        return self._src.getName() + "->" + self._dest.getName()

class WeightedEdge(Edge):
    '''
    带权重的边
    '''
    def __init__(self, src, dest, weight=1.0):
        """
        :param float weight:权重
        """
        Edge.__init__(self, src, dest)
        self._weight = weight

    def getWeight(self):
        return self._weight

    def __str__(self):
        return self._src.getName() + "->(" +str(self._weight) + ")"\
                + self._dest.getName()

class Digraph(object):
    '''
    有向图
    '''
    def __init__(self):
        #保存每一个节点
        self._nodes = []
        #保存节点与其他节点间的联系
        self._edges = {}
    def addNode(self, node):
        '''
        添加节点
        :param Node node:节点
        '''
        if node in self._nodes:
            raise ValueError("Deplicate node.")
        self._nodes.append(node)
        self._edges[node] = []

    def addEdge(self, edge):
        '''
        添加边，建立节点与节点之间的关联
        :param (Edge) edge：边
        '''
        src = edge.getSource()
        dest = edge.getDestination()
        if src not in self._nodes or dest not in self._nodes:
            raise ValueError("Node not in graph.")
        self._edges[src].append(dest)

    def childrenOf(self, node):
        '''
        获得节点的子节点
        :param (Node) node:节点
        :return （list)：所有子节点
        '''
        if node not in self._nodes:
            raise ValueError("Node not in graph.")
        return self._edges[node]

    def hasNode(self, node):
        '''
        判断图中是否含有此节点
        :param (Node) node:节点
        :return bool:是否
        '''
        return node in self._nodes

    def __str__(self):
        result = ''
        for src in self._nodes:
            for dest in self._edges[src]:
                result += src.getName() + '->' + dest.getName() + '\n'
        return result[:-1]

class Graph(Digraph):
    '''
    无向图
    '''
    def addEdge(self, edge):
        '''
        重写函数
        '''
        Digraph.addEdge(self, edge)
        rev = Edge(edge.getDestination(), edge.getDestination())
        Digraph.addEdge(self, rev)

#深度优先搜索算法（DFS)
def printPath(path):
    '''
    打印路径
    :param (list) path:路径的节点列表
    :return (string) :返回路径的转化字符串
    '''
    result = ''
    for i in range(len(path)):
        result = result + str(path[i])
        if i != len(path) - 1:
            result += '->'
    return result

def DFS(graph, start, end, path, shortest):
    '''
    深度优先搜索算法
    :param graph:有向图
    :param start:开始节点
    :param end: 结束节点
    :param path: 当前路径
    :param shortest: 最短路径
    :return ：返回从开始节点到结束节点的最短路径
    '''
    path = path+[start]
    print("Current DFS path:", printPath(path))
    if start == end:
        return path
    for node in graph.childrenOf(start):
        if node not in path:
            if shortest is None or len(path) < len(shortest):
                newPath = DFS(graph, node, end, path, shortest)
                if newPath is not None:
                    shortest = newPath
    return shortest

def search(graph, start, end):
    '''
    搜索有向图的最短路径
    :param start:开始节点
    :param end: 结束节点
    :return ：返回从开始节点到结束节点的最短路径
    '''
    return DFS(graph, start, end, [], None)

def BFS(graph, start, end):
    '''
    广度优先搜索算法
    :param graph:有向图
    :param start:开始节点
    :param end: 结束节点
    '''
    initPath = [start]
    pathQueue = [initPath]
    while len(pathQueue) != 0:
        #删除并获取pathQueue中最早的路径
        tmpPath = pathQueue.pop(0)
        print('Current BFS path:', printPath(tmpPath))
        lastNode = tmpPath[-1]
        if lastNode == end:
            return tmpPath
        for node in graph.childrenOf(lastNode):
            if node not in tmpPath:
                newPath = tmpPath + [node]
                pathQueue.append(newPath)
    return None

def testSP():
    '''
    测试深度搜索算法
    '''
    nodes = []
    for name in range(6):
        nodes.append(Node(str(name)))
    g = Digraph()
    for n in nodes:
        g.addNode(n)
    g.addEdge(Edge(nodes[0], nodes[1]))
    g.addEdge(Edge(nodes[1], nodes[2]))
    g.addEdge(Edge(nodes[2], nodes[3]))
    g.addEdge(Edge(nodes[2], nodes[4]))
    g.addEdge(Edge(nodes[3], nodes[4]))
    g.addEdge(Edge(nodes[3], nodes[5]))
    g.addEdge(Edge(nodes[0], nodes[2]))
    g.addEdge(Edge(nodes[1], nodes[0]))
    g.addEdge(Edge(nodes[3], nodes[1]))
    g.addEdge(Edge(nodes[4], nodes[0]))
    sp = search(g, nodes[0], nodes[5])
    print("Shortest path found by DFS:", printPath(sp))

    sp = BFS(g, nodes[0], nodes[5])
    print("Shortest path found by BFS:", printPath(sp))

if __name__ == "__main__":
    '''
    n1 = Node('a')
    n2 = Node('b')
    n3 = Node('c')
    edge = Edge(n1, n2)
    edge2 = Edge(n3, n2)
    d = Digraph()
    d.addNode(n1)
    d.addNode(n2)
    d.addNode(n3)
    d.addEdge(edge)
    d.addEdge(edge2)
    print(d)
    '''

    testSP()
    a = [1]
    print(id(a))
    a = a + [2]
    print(id(a))


