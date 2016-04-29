# -*- coding:utf-8 -*-
'''
author: Ye Hu
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(figsize=(15,10))
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
ax.set_axis_off()
[x, t] = np.meshgrid(np.array([range(25)])/24.0, np.arange(0, 575.5, 0.5)/575*17*np.pi-2*np.pi)
p = (np.pi/2)*np.exp(-t/(8*np.pi))
u = 1-(1-np.mod(3.6*t, 2*np.pi)/np.pi)**4/2
y = 2*(x**2-x)**2*np.sin(p)
r = u*(x*np.sin(p)+y*np.cos(p))

surf = ax.plot_surface(r*np.cos(t), r*np.sin(t), u*(x*np.cos(p)-y*np.sin(p)), rstride=1, cstride=1, cmap=cm.gist_rainbow_r,
                        linewidth=0, antialiased=True)

plt.savefig('flower.png', dpi=100)
plt.show()

'''
id  铸件编号 工序  工位  工位定额
1    H01      1   车    12
2    H01      2   车    13
3    H01      3   洗    24
'''