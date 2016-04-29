# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
def isSubset(L1, L2):
    '''
    判断列表L1是否是L2的子集
    '''
    for e1 in L1:
        matched = False
        for e2 in L2:
            if e1 == e2:
                matched = True
                break
        if not matched:
            return False
    return True
def intersect(L1, L2):
    '''
    返回L1与L2中的公共元素
    '''
    results = []
    for e1 in L1:
        for e2 in L2:
            if e1 == e2:
                if e1 not in results:
                    results.append(e1)
    return results

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

if __name__ == "__main__":
    print(isSubset([1, 2], [2, 1, 3]))
    print(isSubset([2, 3], [1, 3, 4]))
    print(intersect([1, 2, 3], [3, 2, 4]))
    print(genPowerset(['1', 'b', 'v']))