# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
def findDivisors(n1, n2):
    '''
    返回n1与n2的公约数，以元组方式返回
    '''
    divisors = tuple()
    for i in range(2, min(n1, n2)+1):
        if n1%i == 0 and n2%i == 0:
            divisors += (i,)
    return divisors

def findExtremeDivisors(n1, n2):
    '''
    返回一个元组，分别保存n1与n2的最小公约数与最大公约数
    '''
    minVal, maxVal = None, None
    for i in range(2, min(n1, n2)+1):
        if n1%i == 0 and n2%i == 0:
            if minVal is None or i < minVal:
                minVal = i
            if maxVal is None or i > maxVal:
                maxVal = i
    return minVal, maxVal

def removeDups(L1, L2):
    for e1 in L1:
        print(e1)
        if e1 in L2:
            L1.remove(e1)
from math import hypot
class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return "Vector({0}, {1})".format(self.x, self.y)
    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("{0} must be a vector".format(other))
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    def __mul__(self, scalar):
        if not isinstance(scalar, int):
            raise TypeError("{0} must be a interger".format(scalar))
        return Vector(self.x*scalar, self.y*scalar)
    def __abs__(self):
        return hypot(self.x, self.y)
    def __bool__(self):
        return bool(abs(self))

class Person(object):
    "Person类"
    a = ('Yehu')
    def __init__(self, name, age, gender):
        print('进入Person的初始化')
        self.name = name
        self.age = age
        self.gender = gender
        print('离开Person的初始化')
    def getName(self):
        print(self.name)

if __name__ == "__main__":
    p = Person('ice', 18, '男')
    print(Person.__dict__)
    print(p.__dict__)
    p.weight = '70kg'
    print(p.__dict__)
