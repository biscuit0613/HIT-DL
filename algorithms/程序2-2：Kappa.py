# -*- coding: utf-8 -*-
from sklearn.metrics import cohen_kappa_score
#kappa相关系数和pearson相关系数的区别。

rater1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rater2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
print ("kappa correlation coefficent:",cohen_kappa_score(rater1, rater2))

from scipy.stats import pearsonr
print ("pearson correlation coefficent:", pearsonr(rater1, rater2))

