# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
"""
谎言与统计
"""
import matplotlib.pylab as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['lines.linewidth'] = 2.5

def plotHousing(impression):
    """
    生成房价随时间变化的图标
    """
    f = open("midWestHousingPrices.txt", 'r')
    #文件每一行是年季度价格
    labels, prices = [], []
    for line in f:
        year, quarter, price = line.split()
        label = year[2:4] + "\n Q" + quarter[1]
        labels.append(label)
        prices.append(float(price)/1000)
    #柱的X坐标
    quarters = np.arange(len(labels))
    #柱宽
    width = 0.5
    if impression == 'flat':
        plt.semilogy()
    plt.bar(quarters, prices, width, color='r')
    plt.xticks(quarters + width / 2.0, labels)
    plt.title("美国中西部各州房价")
    plt.xlabel("季度")
    plt.ylabel("平均价格($1000)")

    if impression == 'flat':
        plt.ylim(10, 10**3)
    elif impression == "volatile":
        plt.ylim(180, 220)
    elif impression == "fair":
        plt.ylim(150, 250)
    else:
        raise ValueError("Invalid input.")



if __name__ == "__main__":
    plt.figure()
    plotHousing("flat")
    plt.savefig("chapter16_3.jpg", dpi=100)
    plt.show()