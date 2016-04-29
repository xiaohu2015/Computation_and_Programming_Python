# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
from functools import wraps
import re
#判断str1是否是str2的字串
def is_subStr(str1, str2):
    return str1 in str2
def printName(firstName, lastName, reverse=False):
    if reverse:
        print(lastName+',', firstName)
    else:
        print(firstName, lastName)
#寻找根
def findRoot(x, power, epsilon=0.0001):
    '''
    x: 要求根的数，整数或者浮点数；
    power:>=1的整数，指数
    epsilon：精度，较小的浮点数
    return：如果无根，返回None
    '''
    #负数无偶次根
    if x < 0 and power%2 == 0:
        return None
    low = min(-1.0, x)
    high = max(1.0, x)
    root = (low+high)/2
    while abs(root**power - x) >= epsilon:
        if root**power < x:
            low = root
        else:
            high = root
        root = (low+high)/2
    return root
#测试函数
def testFindRoot():
    for x in (0.25, -0.25, 2, -2, 8, -8):
        for power in range(1, 5):
            print('Testing x =', x, "and power =", power)
            result = findRoot(x, power)
            if result is None:
                print("No root")
            else:
                print("Root:", result)

def factI(n):
    '''
    阶乘 循环
    '''
    result = 1
    while n > 1:
        result = result*n
        n = n - 1
    return result

def factR(n):
    '''
    阶乘 递归
    '''
    if n == 1:
        return 1
    return n*factR(n-1)
def counter(n):
    num = 0
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal num
            if n in args:
                num += 1
                print(num)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@counter(2)
def fib(n):
    '''
    斐波那契数列, 递归
    '''
    if n == 0 or n == 1:
        return 1
    return fib(n-1) + fib(n-2)

def isPalindrome(s):
    '''
    判断s是否是回文字符串，忽略标点符号，空格，大小写
    '''
    def toChars(s):
        '''
        将一个字符串中非字符符号剔除
        '''
        letters = ''
        for c in map(lambda s: s.lower(), s):
            if c in 'abcdefghijklmnopqrstuvwxyz':
                letters += c
        return letters
    #判断是都是回文字符串
    def isPal(s):
        if len(s) <= 1:
            return True
        return s[0] == s[-1] and isPal(s[1:-1])
    return isPal(toChars(s))

if __name__ == "__main__":
    print(isPalindrome('do23goD'))