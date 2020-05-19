# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:08:58 2020

@author: wmy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import string
import category_encoders as ce
import time

rawtrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/train.csv')
rawtest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/test.csv')
sub=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sample_submission.csv')
target=rawtrain['target']

train=rawtrain.drop(['id','target'],axis=1)
test=rawtest.drop('id',axis=1)

# TE=ce.TargetEncoder(cols=te_list, min_samples_leaf=1, smoothing=0.3)
# train.loc[:,te_list]=TE.fit_transform(train,target)
# test.loc[:,te_list]=TE.transform(test)
#======encode ordinal
cate_ord=['ord_1','ord_2']
for c in cate_ord:
    print(rawtrain[c].unique())
levelmap={c:i for i,c in enumerate(['Novice','Contributor', 'Expert', 'Master','Grandmaster'])}
train['ord_1']=train['ord_1'].replace(levelmap)
test['ord_1']=test['ord_1'].replace(levelmap)
tempratmap={c:i for i,c in enumerate(['Freezing','Cold', 'Warm','Hot' , 'Boiling Hot' ,'Lava Hot' ])}
train['ord_2']=train['ord_2'].replace(tempratmap)
test['ord_2']=test['ord_2'].replace(tempratmap)
lowermap={c:i for i,c in enumerate(string.ascii_lowercase)}
train['ord_3']=train['ord_3'].replace(lowermap)
test['ord_3']=test['ord_3'].replace(lowermap)
upperletter=rawtrain['ord_4'].unique().tolist()
upperletter.remove(np.nan)
upperletter.sort()
uppermap={c:i for i,c in enumerate(string.ascii_uppercase)}
train['ord_4']=train['ord_4'].replace(uppermap)
test['ord_4']=test['ord_4'].replace(uppermap)
#/ord_5
alletter=string.ascii_letters
allmap={c:i for i,c in enumerate(alletter)}
def getP(x,p):
    if pd.isnull(x):
        return x
    else:
        if p==0:
            return x[0]
        else:
            return x[1]
        
train['ord_5_0']=rawtrain['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)
train['ord_5_1']=rawtrain['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)
test['ord_5_0']=rawtest['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)
test['ord_5_1']=rawtest['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)
train=train.drop('ord_5',axis=1)
test=test.drop('ord_5',axis=1)
te_list=train.columns.tolist()
def TargetEncode(trainc,testc,targetc, smooth):
    print('Target encoding...')
    smoothing=smooth
    oof = np.zeros(len(trainc))
    for tr_idx, oof_idx in StratifiedKFold(n_splits=10, random_state=2020, shuffle=True).split(trainc, targetc):
        train_x=trainc.iloc[tr_idx].reset_index(drop=True)
        valid_x=trainc.iloc[oof_idx].reset_index(drop=True)
        target_train=targetc.iloc[tr_idx].reset_index(drop=True)
        prior = target_train.mean()
        tmp = target_train.groupby(train_x).agg(['sum', 'count'])
        tmp['mean'] = tmp['sum'] / tmp['count']
        smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))
        cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 
        tmp['smoothing'] = cust_smoothing
        tmp = tmp['smoothing'].to_dict()
        oof[oof_idx]=valid_x.map(tmp).astype(float).fillna(prior).values
    prior = targetc.mean()
    tmp = targetc.groupby(trainc).agg(['sum', 'count'])
    tmp['mean'] = tmp['sum'] / tmp['count']
    smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))
    cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 
    tmp['smoothing'] = cust_smoothing
    tmp = tmp['smoothing'].to_dict()
    testc=testc.map(tmp).astype(float).fillna(prior)
    return oof, testc

for n in te_list:
    train[n],test[n]=TargetEncode(train[n],test[n],target,0.3)
#==========boost tree MODEL======================

usedfeatures=test.columns.tolist()#bicolnames+norcolnames+[n+'_miss' for n in normcol59]+[n+'_target' for n in normcol59]+['ord_1','ord_2','ord_3','ord_4','ord_5']
# cat_cols=['cluster']#+[n+'_freq' for n in normcol59]#+['ord_1','ord_2','ord_3','ord_4','ord_5']+[n+'_miss' for n in normcol59]+bicolnames+norcolnames+
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1990)

t1=time.clock()
traintion = np.zeros(len(train))
validation = np.zeros(len(train))
predictions = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train.iloc[trn_idx][usedfeatures].reset_index(drop=True)
    valid_x=train.iloc[val_idx][usedfeatures].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)


    GB=GaussianNB()
    GB.fit(train_x,target_train)
    traintion[trn_idx] += GB.predict_proba(train_x)[:,1]/(folds.n_splits-1)
    validation[val_idx] = GB.predict_proba(valid_x)[:,1]
    

    predictions += GB.predict_proba(test[usedfeatures])[:,1] / folds.n_splits
t2=time.clock()-t1
print("Train AUC score: {:<8.5f}".format(roc_auc_score(target,traintion)))
print("Valid AUC score: {:<8.5f}".format(roc_auc_score(target,validation)))
#10cv
# Train AUC score: 0.76030 
# Valid AUC score: 0.76021 
# 20cv
# Train AUC score: 0.76029 
# Valid AUC score: 0.76020 


 
sub['target']=predictions
pd.Series(validation).to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/gb_10cv_targetencode_cluster_validation.csv',index=False,header=False)
sub.to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/gb_10cv_targetencode_cluster.csv',index=False)
