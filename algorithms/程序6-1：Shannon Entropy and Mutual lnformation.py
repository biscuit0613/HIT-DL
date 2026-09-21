# -*- coding: utf-8 -*-
#演示内容：香农信息熵的计算(例1和例2分别为两种不同类型的输入)以及互信息的计算（例3）。其中log默认为自然对数。

import numpy as np
from math import log

#例1： 计算香农信息熵（已知概率分布）
print("例1：") 
def calc_ent(x):   
    ent = 0.0
    for p in x:
        ent -= p * np.log(p)
    return ent

x1=np.array([0.4, 0.2, 0.2, 0.2])
x2=np.array([1])
x3=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
print ("x1的信息熵:", calc_ent(x1))
print ("x2的信息熵:", calc_ent(x2))
print ("x3的信息熵:", calc_ent(x3))
print("") 

#例2： 计算香农信息熵（此时给定了信号发生情况) 
print("例2：") 
def calcShannonEnt(dataSet):  
    length,dataDict=float(len(dataSet)),{}  
    for data in dataSet:  
        try:dataDict[data]+=1  
        except:dataDict[data]=1  
    return sum([-d/length*log(d/length) for d in list(dataDict.values())])  
print("x1的信息熵:", calcShannonEnt(['A','B','C','D','A'])) 
print("x2的信息熵:",calcShannonEnt(['A','A','A','A','A'])) 
print("x3的信息熵:",calcShannonEnt(['A','B','C','D','E'])) 


#例3： 计算互信息（输入：给定的信号发生情况,其中联合分布已经手工给出)
print("") 
print("例3：") 
Ent_x4=calcShannonEnt(['3',  '4',   '5',  '5', '3',  '2',  '2', '6', '6', '1'])
Ent_x5=calcShannonEnt(['7',  '2',   '1',  '3', '2',  '8',  '9', '1', '2', '0'])
Ent_x4x5=calcShannonEnt(['37', '42', '51', '53', '32', '28', '29', '61', '62', '10', '37', '42'])
MI_x4_x5=Ent_x4+Ent_x5-Ent_x4x5
print ("x4和x5之间的互信息:",MI_x4_x5)

