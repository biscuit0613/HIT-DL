# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:42:47 2022

@author: lenovo
"""
critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,  
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,  
                         'The Night Listener': 3.0},  
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,  
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,  
                            'You, Me and Dupree': 3.5},  
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,  
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},  
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,  
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,  
                            'You, Me and Dupree': 2.5},  
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,  
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,  
                            'You, Me and Dupree': 2.0},  
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,  
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},  
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}  
  
  
from math import sqrt  
  
def sim_pearson(prefs, p1, p2):  
    # Get the list of mutually rated items  
    si = {} # 初始化一个空字典，用于存储两个用户都评分过的电影  
    for item in prefs[p1]:  # 遍历第一个用户评分过的电影
        if item in prefs[p2]:  # 如果第二个用户也评分过该电影
            si[item] = 1  # 将该电影添加到字典si中，值为1，表示两个用户都评分过该电影
  
    # if they are no ratings in common, return 0  
    if len(si) == 0:  
        return 0  
  
    # Sum calculations  
    n = len(si)  
  
    # Sums of all the preferences  
    sum1 = sum([prefs[p1][it] for it in si])  # prefs[p1][it]表示第一个用户对电影it的评分，sum1是第一个用户对所有共同评分电影的评分总和
    sum2 = sum([prefs[p2][it] for it in si])  # prefs[p2][it]表示第二个用户对电影it的评分，sum2是第二个用户对所有共同评分电影的评分总和
  
    # Sums of the squares  
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])  
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])  
  
    # Sum of the products  
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])  
  
    # Calculate r (Pearson score)  
    num = pSum - (sum1 * sum2 / n)  
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))  
    if den == 0:  
        return 0  
  
    r = num / den  
  
    return r  
  
print(sim_pearson(critics,'Lisa Rose','Gene Seymour'))  
#0.396059017191  