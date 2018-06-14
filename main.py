# -*- coding: utf-8 -*-
"""
Created on Mon May 28 05:43:15 2018

@author: Rahul
"""
import joblib
import os, argparse
import pandas as pd
import numpy as np
os.chdir("C:/Users/Rahul/Desktop/edwisor")
import importing


class health(object):
	def getLoadOption(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--Data_File', action='store', dest='Data_File')
		
		self.result_op = parser.parse_args()
		
		return self.result_op
		
def main():
    cli = health()
    cli_line= cli.getLoadOption()
    data_file = cli_line.Data_File
    pred_data=pd.read_csv(data_file)
    vec = importing.vec
    diction=joblib.load('diction.pkl')
    label_columns=importing.label_cols
    preds = np.zeros((len(pred_data), len(label_columns)))
    test_term_doc = vec.transform(pred_data['body'])
    for i,j in enumerate(label_columns):
            m=diction[j][0]
            r=diction[j][1]
            preds[:,i] = m.predict_proba(test_term_doc.multiply(r))[:,1]

    predictions=pd.DataFrame(preds, columns = label_columns)
    
    primary_prob=[]
    secondary_prob=[]
    for i in range(0,len(predictions)):
        p=[]
        s=[]
        for j in (label_columns):
            if predictions[j][i] >= 0.4:
                p.append(j)
            elif (predictions[j][i] > 0.005 and predictions[j][i] < 0.4 ):
                s.append(j)
        secondary_prob.append(s)
        if len(p)==0:
            primary_prob.append(s)
        else :
            primary_prob.append(p)

    dictionary={'body':list(pred_data['body'].values),'Primary_complains':primary_prob,'Secondary_complains':secondary_prob}
    submission=pd.DataFrame.from_dict(dictionary)
    submission=submission[['body','Primary_complains','Secondary_complains']]
    submission.to_csv('final_report.csv',index=False)
if __name__ == "__main__":
	main()



    
