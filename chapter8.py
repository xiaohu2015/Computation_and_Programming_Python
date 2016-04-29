# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
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

def compareMortgages(amt, years, fixedRate, pts, ptsRate, varRate1, varRate2, varMonths):
    totMonths = years*12
    fixed1 = Fixed(amt, fixedRate, totMonths)
    fixed2 = FixedWithPts(amt, ptsRate, totMonths, pts)
    twoRate = TwoRate(amt, varRate2, totMonths, varRate1, varMonths)
    morts = [fixed1, fixed2, twoRate]
    for m in range(totMonths):
        for mort in morts:
            mort.makePayment()
    for m in morts:
        print(m)
        print(" Total payments = $"+str(int(m.getTotalPaid())))


if __name__ == "__main__":
    compareMortgages(amt=200000, years=30, fixedRate=0.07, pts=3.25, ptsRate=0.05, varRate1=0.045,
                     varRate2=0.095, varMonths=48)