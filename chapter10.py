# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
import random
def search(L, e):
    """
    假定L中元素按升序排列，如果e在L中,就返回true,否则就返回false
    """
    for x in L:
        if e == x:
            return True
        if e < x:
            return False
    return False

def dichotomySearch(L, e):
    """
    假定L中元素按升序排列，如果e在L中,就返回true,否则就返回false,采用二分法
    """
    def bSearch(L, e, low, high):
        if low == high:
            return L[low] == e
        else:
            mid = (low+high)//2
            if L[mid] == e:
                return True
            elif L[mid] > e:
                if low == mid:
                    return False
                else:
                    return bSearch(L, e, low, mid-1)
            else:
                return bSearch(L, e, mid+1, high)


    if len(L) == 0:
        return False
    else:
        return bSearch(L, e, 0, len(L)-1)

#选择排序
def selSort(L):
    """
    假定L是列表，元素可以用>进行比较，用升序排列L中的元素
    """
    for i in range(0, len(L)):
        for j in range(i+1, len(L)):
            if L[i] > L[j]:
                L[i], L[j] = L[j], L[i]

def merge(left, right, compare):
    """
    是已经排序的列表，定义了元素的顺序，返回一个新的有序列表，包含left与right中的所有元素
    """
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if compare(left[i], right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    return result
import operator
def mergeSort(L, compare=operator.lt):
    """
    假定L是列表,compare定义了元素的顺序
    返回一个排序后的列表
    """
    if len(L) < 2:
        return L[:]
    else:
        mid = len(L)//2
        left = mergeSort(L[:mid], compare)
        right = mergeSort(L[mid:], compare)
        return merge(left, right, compare)

def lastNameFirstName(name1, name2):
    '''
    比较两个名字，先比较姓，然后是名
    '''
    names1 = name1.split(' ')
    names2 = name2.split(' ')
    if names1[1] == names2[1]:
        return names1[0] < names2[0]
    else:
        return names1[1] < names2[1]

def firstNameLastName(name1, name2):
    '''
    比较两个名字，先比较名字，然后是姓
    '''
    names1 = name1.split(" ")
    names2 = name2.split(' ')
    if names1[0] == names2[0]:
        return names1[1] < names2[1]
    else:
        return names1[0] < names2[0]

class IntDict(object):
    """
    键值为整数的字典
    """
    def __init__(self, numBuckets):
        """
        创建一个空字典
        """
        self.__buckets = []
        self.numBuckets = numBuckets
        for i in range(numBuckets):
            self.__buckets.append([])

    @property
    def buckets(self):
        return self.__buckets

    def addEntry(self, dictKey, dictVal):
        hashBucket = self.__buckets[dictKey % self.numBuckets]
        for i in range(len(hashBucket)):
            if hashBucket[i][0] == dictKey:
                hashBucket[i] = (dictKey, dictVal)
                return
        hashBucket.append((dictKey, dictVal))

    def getValue(self, dictKey):
        """
        假定dictKey是整数，返回对应的值
        """
        hashBucket = self.__buckets[dictKey%self.numBuckets]
        for e in hashBucket:
            if e[0] == dictKey:
                return e[1]
        return None

    def __str__(self):
        """
        返回字符串形式
        """
        result = '{'
        for b in self.__buckets:
            for e in b:
                result += str(e[0]) + ':' + str(e[1]) + ','
        return result[:-1] + '}'



if __name__ == "__main__":
    print(dichotomySearch([1, 2, 3, 6, 7], 5))
    L = [3, 10, 12, 2, 6, 10, 6, 8]
    #selSort(L)
    print(mergeSort(L))

    L2 = ['Chris Terman', 'Tom Brady', 'Eric Grimson', 'Gisele Bundchen']
    print("Sorted by last name =", mergeSort(L2, lastNameFirstName))
    print("Sorted by first name =", mergeSort(L2, firstNameLastName))
    D = {'a':12, 'c': 5, 'b': 14}
    print(sorted(D.items(), key= lambda x : x[1], reverse=True))
    L3 = [[1, 2, 3], (3, 2, 1, 0), 'abc']
    print(sorted(L3, key=len, reverse=True))

    D = IntDict(40)
    for i in range(40):
        key = random.randint(0, 10**5)
        D.addEntry(key, i)

    print('The value of the intDict is: ')
    print(D)
    print("\n, The buckets are:")
    for hashBucket in D.buckets:
        print("   ", hashBucket)

