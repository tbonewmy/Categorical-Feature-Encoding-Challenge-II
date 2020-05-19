# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:23:58 2020

@author: wmy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
import multiprocessing as multip
from joblib import Parallel, delayed

num_cores = multip.cpu_count()
rawtrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/train.csv')
sub=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sample_submission.csv')
target=rawtrain['target']

gbTrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/gb_10cv_targetencode_cluster_validation.csv')
gbTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/gb_10cv_targetencode_cluster.csv')

cbTrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/cb_40cv_labelencode_cluster_validation.csv')
cbTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/cb_40cv_labelencode_cluster.csv')

lrTrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lr_10cv_onehotall_validation.csv')
lrTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lr_10cv_onehotall.csv')

bbTrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/bb_40cv_onehotall_validation.csv')
bbTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/bb_40cv_onehotall.csv')

# lgbTrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lgb_10cv_param_LeOd56789_FeqNo_10cvTeNoBi_2ord5_allmiss_cluster_validation.csv')
# lgbTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lgb_10cv_param_LeOd56789_FeqNo_10cvTeNoBi_2ord5_allmiss_cluster.csv')
lgbTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/submission.csv')

krTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/keras-score.csv')

ctTest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/catboost-score.csv')

p1=np.asarray(range(0,11))*0.1
p2=np.asarray(range(0,11))*0.1
p3=np.asarray(range(0,11))*0.1
p4=np.asarray(range(0,11))*0.1
p5=np.asarray(range(0,11))*0.1

def getAUC(a,b,c,d,e):
    if a+b+c+d+e==1:
        print('{}{}{}{}{}'.format(a,b,c,d,e),flush=True)
        pred=gbTrain.iloc[:,0]*a+cbTrain.iloc[:,0]*b+lrTrain.iloc[:,0]*c+bbTrain.iloc[:,0]*d+lgbTrain.iloc[:,0]*e
        return (roc_auc_score(target[1:],pred),[a,b,c,d,e])

if __name__ == '__main__':
    poolv = multip.Pool(num_cores) 
    auc=[]
    params=[]
    alldata=[]
    for a in p1:
        for b in p2:
            for c in p3:
                for d in p4:
                    print('{}{}{}{}'.format(a,b,c,d),flush=True)
                    outcomes = Parallel(n_jobs=num_cores)(delayed(getAUC)(a,b,c,d,e) for e in p5)
                    alldata+=outcomes
                    # for e in p5:
                    #     if a+b+c+d+e==1:
                    #         pred=gbTrain.iloc[:,0]*a+cbTrain.iloc[:,0]*b+lrTrain.iloc[:,0]*c+bbTrain.iloc[:,0]*d+lgbTrain.iloc[:,0]*e
                    #         auc.append(roc_auc_score(target[1:],pred))
                    #         params.append([a,b,c,d,e])
auc_list=[r for r in alldata if r!= None]
auc_list_sorted=sorted(auc_list,key=lambda x:x[0],reverse=True)
auc_list_sorted[1][1]

sub['target']=cbTest.target*0.05+lrTest.target*0.1+lgbTest.target*0.1+krTest.target*0.35+ctTest.target*0.4
sub.to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/blend005010103504_newlgb.csv',index=False)
