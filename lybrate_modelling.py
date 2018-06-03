# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:21:26 2018

@author: Rahul
"""

import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
os.chdir("C:/Users/Rahul/Desktop/edwisor")
import importing

x = importing.trn_term_doc
label_cols = importing.label_cols
train_data=importing.train_data
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

dict1={}
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train_data[j])
    dict1.update({j:[m,r]})
joblib.dump(dict1,'diction.pkl')







