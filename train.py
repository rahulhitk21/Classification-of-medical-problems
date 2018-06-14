# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:43:51 2018

@author: Rahul
"""

import pandas as pd
import os
import numpy as np


os.chdir("C:/Users/Rahul/Desktop/edwisor")
raw_data=pd.read_csv("lybrate_ml_test.csv")
collist=list(raw_data.columns)
for i in collist:
   raw_data[i]=raw_data[i].astype(str)
    
sm=[]
for i in range(0,len(raw_data)):
    m=[]
    m=raw_data['secondary_complain'][i].split(',')
    for j in m:
        k=j[1:len(j)]
        w=k[1:len(k)]
        g=w.strip("'")
        sm.append(g)
        
for i in range(0,len(raw_data)):
    m=[]
    m=raw_data['primary_complain'][i].split(',')
    for j in m:
        k=j[1:len(j)]
        w=k[1:len(k)]
        g=w.strip("'")
        sm.append(g)
collist=list(set(sm))
collist.remove('')     
collist=[x.lower() for x in collist]
collist=list(set(collist))
new_data=pd.DataFrame(0,np.arange(len(raw_data)),columns=collist)
trans_data=raw_data.join(new_data)
trans_data=trans_data.drop('secondary_complain', axis=1)
trans_data=trans_data.drop('primary_complain', axis=1)

for i in range(0,len(raw_data)):
    m=[]
    m=raw_data['secondary_complain'][i].split(',')
    for j in m:
        k=j[1:len(j)]
        w=k[1:len(k)]
        g=w.strip("'")
        g=g.lower()
        if g in collist:
            trans_data[g][i]=1
for i in range(0,len(raw_data)):
    m=[]
    m=raw_data['primary_complain'][i].split(',')
    for j in m:
        k=j[1:len(j)]
        w=k[1:len(k)]
        g=w.strip("'")
        g=g.lower()
        if g in collist:
            trans_data[g][i]=1

trans_data.to_csv('trans_final_data_local_afterpm1.csv',index=False) 

        
