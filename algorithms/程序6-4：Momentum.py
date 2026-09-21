# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:46:28 2021
https://github.com/hsmyy/zhihuzhuanlan/blob/master/momentum.ipynb
https://zhuanlan.zhihu.com/p/21486826
路遥知马力——Momentum
@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x[0] * x[0] + 60 * x[1] * x[1]
def g(x):
    return np.array([2 * x[0], 100 * x[1]])
xi = np.linspace(-220,220,1000)
yi = np.linspace(-100,100,1000)
X,Y = np.meshgrid(xi, yi)
Z = X * X + 20 * Y * Y


def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(15, 8))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel("w2", rotation=0, fontsize=30)
    plt.xlabel("w1", fontsize=30)
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*', markersize=27.0)
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i+2, 0], arr[i:i+2, 1], '-g', linewidth=3.0)
        
contour(X,Y,Z)



def gd(x_start, step, g):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step
        
        passing_dot.append(x.copy())
        print ('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot
res, x_arr = gd([-180,75], 0.016, g)
contour(X,Y,Z, x_arr)


res, x_arr = gd([-180,75], 0.019, g)
contour(X,Y,Z, x_arr)

res, x_arr = gd([-180,75], 0.02, g)
contour(X,Y,Z, x_arr)

def momentum(x_start, step, g, discount = 0.7):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        
        passing_dot.append(x.copy())
        print ('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot
res, x_arr = momentum([-180,75], 0.016, g)
contour(X,Y,Z, x_arr)

def nesterov(x_start, step, g, discount = 0.7):   # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * 0.7 + grad 
        x -= pre_grad * step
        
        passing_dot.append(x.copy())
        print ('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot
res, x_arr = nesterov([-180,75], 0.012, g)
plt.contour(X, Y, Z, colors='black')