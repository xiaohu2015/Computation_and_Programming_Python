# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
def sum_from_string(dataStr):
    dataList = dataStr.split(',')   #生成列表
    sum = 0.0
    for value in map(float, dataList):
        sum += value
    return sum

#穷举法求平方根
def find_square_root(value, epsilon):
    step = epsilon**2
    numGuesses = 0
    root = 0
    while abs(value - root**2) >= epsilon and root**2 <= value+epsilon:
        root += step
        numGuesses += 1
    print('numGuesses:', numGuesses)
    return root
#二分法求平方根
def find_sqRoot_new(value, epsilon=0.01):
    numGuesses = 0
    low = 0.0
    high = max(1.0, value)
    root = (high+low)/2
    while abs(root**2-value) >= epsilon:
        print("low =", low, "high=", high, "root=", root)
        numGuesses += 1
        if root**2 < value:
            low = root
        else:
            high = root
        root = (high+low)/2
    print("numGuesses:", numGuesses)
    return root

#求解任意方根
def find_root(value, epsilon, index):
    if value < 0 and index%2 ==0:
        return "Error"
    numGuesses = 0
    if value < 0:
        low = min(-1.0, value)
        high = 0.0
    else:
        low = 0.0
        high = max(1.0, value)
    root = (high+low)/2
    while abs(root**index-value) >= epsilon:
        print("low =", low, "high=", high, "root=", root)
        numGuesses += 1
        if root**index < value:
            low = root
        else:
            high = root
        root = (high+low)/2
    print("numGuesses:", numGuesses)
    return root

#用牛顿-拉夫逊方法寻找平方根
def find_sRoot_newton(value, epsilon=0.01):
    numGuesses = 0
    guess = value/2.0

    while abs(guess**2-value) >= epsilon:
        guess -= (guess**2-value)/(2*guess)
        numGuesses += 1
    print("numGuesses:", numGuesses)
    return guess


if __name__ == "__main__":
    dataStr = "1.23,2.4,3.123"
    print(sum_from_string(dataStr))
    print(find_sqRoot_new(4500000))
    print(find_sRoot_newton(4500000))