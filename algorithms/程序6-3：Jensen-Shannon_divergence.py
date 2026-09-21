# -*- coding: utf-8 -*-
"""
演示内容：KL散度和JS散度
"""
import numpy as np
import scipy.stats
 
p=np.asarray([0.65,0.25,0.07,0.03])
q=np.array([0.6,0.25,0.1,0.05])

#KL散度是不对称的
kl1=np.sum(p*np.log(p/q))
print (kl1)
kl2=np.sum(q*np.log(q/p))
print (kl2)

M=(p+q)/2

#JS散度的计算方法一：根据公式计算
js1=0.5*np.sum(p*np.log(p/M))+0.5*np.sum(q*np.log(q/M))
print (js1)

#JS散度的计算方法二：调用scipy包计算
js2=0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)
print  (js2)