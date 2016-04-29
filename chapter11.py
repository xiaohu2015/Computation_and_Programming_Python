# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
绘图以及类的扩展内容
"""
import pylab
import matplotlib.pyplot as plt
import numpy as np

'''
plt.figure(1)
plt.plot([1, 2, 3, 4], [1, 7, 3, 5])
plt.show()
'''
#设置线条宽度
plt.rcParams['lines.linewidth'] = 3
#设置标题字体大小
plt.rcParams['axes.titlesize'] = 10
#设置坐标轴标签字体大小
plt.rcParams['axes.labelsize'] = 10
#设置X轴数字的大小
plt.rcParams['xtick.labelsize'] = 10
#设置Y轴数字的大小
plt.rcParams['ytick.labelsize'] = 10
#设置X轴上标记的大小
plt.rcParams['xtick.major.size'] = 4
#设置Y轴上标记的大小
plt.rcParams['ytick.major.size'] = 4
#设置标记大小
plt.rcParams['lines.markersize'] = 8


def figure1():
    #初始化投资
    principal = 10000
    interestRate = 0.05
    years = 20
    values = []
    for i in range(years+1):
        values.append(principal)
        principal += principal*interestRate
    plt.plot(values)
    plt.title('5% Growth, Compounded Annually')
    plt.xlabel('Years of Compounding')
    plt.ylabel('Value of principal ($)')

def findPayment(loan, r, m):
    '''
    返回贷款数为loan，每月利率为r，共m个月情况下的每月还款数
    '''
    return loan*((r*(1+r)**m)/((1+r)**m-1))

class Mortgage(object):
    '''
    用来构建不同类型抵押贷款的抽象类
    '''
    def __init__(self, loan, annRate, months):
        '''
        创建一个抵押贷款
        '''
        self.loan = loan
        self.rate = annRate/12
        self.months = months
        self.paid = [0.0]
        self.owed = [loan]
        self.payment = findPayment(loan, self.rate, months)
        self.legend = None

    def makePayment(self):
        '''
        还款
        '''
        self.paid.append(self.payment)
        reduction = self.payment - self.owed[-1]*self.rate
        self.owed.append(self.owed[-1]-reduction)

    def getTotalPaid(self):
        '''
        返回现在总还款数
        '''
        return sum(self.paid)

    def __str__(self):
        return self.legend

    def plotPayments(self, ax, style):
        ax.plot(self.paid[1:], style, label=self.legend)

    def plotBalance(self, ax, style):
        ax.plot(self.owed, style, label=self.legend)

    def plotTotPd(self, ax, style):
        """绘制还款总变化
        """
        totPd = [self.paid[0]]
        for i in range(1, len(self.paid)):
            totPd.append(totPd[-1] + self.paid[i])
        ax.plot(totPd, style, label=self.legend)

    def plotNet(self, ax, style):
        """=绘制抵押贷款的总支出，用现金支出减去通过还清部分贷款所得本金
        """
        totPd = [self.paid[0]]
        for i in range(1, len(self.paid)):
            totPd.append(totPd[-1] + self.paid[i])
        entityAcquired = np.array([self.loan]*len(self.owed))
        entityAcquired = entityAcquired - np.array(self.owed)
        net = np.array(totPd) - entityAcquired
        ax.plot(net, style, label=self.legend)

class Fixed(Mortgage):
    def __init__(self, loan, r, months):
        super().__init__(loan, r, months)
        self.legend = "Fixed, " + str(r*100) + "%"

class FixedWithPts(Mortgage):
    def __init__(self, loan, r, months, pts):
        super().__init__(loan, r, months)
        self.pts = pts
        self.paid = [loan*(pts/100)]
        self.legend = "Fixed, " + str(r*100) + "%, " + str(pts) + " points"

class TwoRate(Mortgage):
    def __init__(self, loan, r, months, teaserRate, teaserMoths):
        super().__init__(loan, teaserRate, months)
        self.teaserMonths = teaserMoths
        self.teaserRate = teaserRate
        self.nextRate = r/12
        self.legend = str(teaserRate*100) + "% for " + str(self.teaserMonths) + " months, then " + str(r*100) + "%"
    def makePayment(self):
        if len(self.paid) == self.teaserMonths + 1:
            self.rate = self.nextRate
            self.payment = findPayment(self.owed[-1], self.rate, self.months-self.teaserMonths)
        Mortgage.makePayment(self)

def plotMortgages(morts, amt):
    styles = ['b-', 'b-.', 'b:']
    payments_ax = plt.subplot(221)
    plt.title("Monthly Payments of Different $" + str(amt) + ' Mortgage')
    plt.xlabel('Months')
    plt.ylabel('Monthly Payments')
    cost_ax = plt.subplot(222)
    plt.title("Cash Outlay of Different $" + str(amt) + ' Mortgage')
    plt.xlabel('Months')
    plt.ylabel('Total Payments')
    balance_ax = plt.subplot(223)
    plt.title("Balance Remaining of $" + str(amt) + ' Mortgage')
    plt.xlabel('Months')
    plt.ylabel('Remaining Loan Balance of $')
    netCost_ax = plt.subplot(224)
    plt.title("Net Cost of $" + str(amt) + ' Mortgage')
    plt.xlabel('Months')
    plt.ylabel('Payments - Equity $')

    for i in range(len(morts)):
        morts[i].plotPayments(payments_ax, styles[i])
        morts[i].plotTotPd(cost_ax, styles[i])
        morts[i].plotBalance(balance_ax, styles[i])
        morts[i].plotNet(netCost_ax, styles[i])
    payments_ax.legend(loc='upper center')
    cost_ax.legend(loc='best')
    balance_ax.legend(loc='best')
def compareMortgages(amt, years, fixedRate, pts, ptsRate, varRate1, varRate2, varMonths):
    totMonths = years*12
    fixed1 = Fixed(amt, fixedRate, totMonths)
    fixed2 = FixedWithPts(amt, ptsRate, totMonths, pts)
    twoRate = TwoRate(amt, varRate2, totMonths, varRate1, varMonths)
    morts = [fixed1, fixed2, twoRate]
    for m in range(totMonths):
        for mort in morts:
            mort.makePayment()
    plotMortgages(morts, amt)



if __name__ == "__main__":
    plt.figure(figsize=(20,10))
    compareMortgages(amt=200000, years=30, fixedRate=0.07, pts=3.25, ptsRate=0.05, varRate1=0.045,
                     varRate2=0.095, varMonths=48)
    plt.plot([1, 2, 2])
    plt.savefig('fig1.png')
    plt.show()

    pass