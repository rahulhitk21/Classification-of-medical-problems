# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:00:45 2018

@author: Rahul
"""
import pandas as pd
import os
import string,re
from sklearn.feature_extraction.text import  TfidfVectorizer

os.chdir("C:/Users/Rahul/Desktop/edwisor")
modelling_data=pd.read_csv('trans_final_data_local_afterpm1.csv', encoding = "ISO-8859-1")
modelling_data['body']=modelling_data['body'].astype(str)
train_data=modelling_data.iloc[:78446,:]
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train_data['body'])
label_cols = list(train_data.drop('body',axis=1).columns)

