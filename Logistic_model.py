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
from sklearn.linear_model import LogisticRegression
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
#train['total_miss']=train.isna().sum(axis=1)
#test['total_miss']=test.isna().sum(axis=1)
for c in train.columns:
    train[c],test[c]=train[c].fillna(train[c].mode()[0]),test[c].fillna(train[c].mode()[0])
traintest = pd.concat([train, test])
dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)
train = dummies.iloc[:train.shape[0], :]
test = dummies.iloc[train.shape[0]:, :]
##======encode ordinal
#cate_ord=['ord_1','ord_2']
#for c in cate_ord:
#    print(rawtrain[c].unique())
#levelmap={c:i for i,c in enumerate(['Novice','Contributor', 'Expert', 'Master','Grandmaster'])}
#train['ord_1']=train['ord_1'].replace(levelmap)
#test['ord_1']=test['ord_1'].replace(levelmap)
#tempratmap={c:i for i,c in enumerate(['Freezing','Cold', 'Warm','Hot' , 'Boiling Hot' ,'Lava Hot' ])}
#train['ord_2']=train['ord_2'].replace(tempratmap)
#test['ord_2']=test['ord_2'].replace(tempratmap)
#lowermap={c:i for i,c in enumerate(string.ascii_lowercase)}
#train['ord_3']=train['ord_3'].replace(lowermap)
#test['ord_3']=test['ord_3'].replace(lowermap)
#upperletter=rawtrain['ord_4'].unique().tolist()
#upperletter.remove(np.nan)
#upperletter.sort()
#uppermap={c:i for i,c in enumerate(string.ascii_uppercase)}
#train['ord_4']=train['ord_4'].replace(uppermap)
#test['ord_4']=test['ord_4'].replace(uppermap)
##/ord_5
#alletter=string.ascii_letters
#allmap={c:i for i,c in enumerate(alletter)}
#def getP(x,p):
#    if pd.isnull(x):
#        return x
#    else:
#        if p==0:
#            return x[0]
#        else:
#            return x[1]
#        
#train['ord_5_0']=train['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)
#train['ord_5_1']=train['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)
#test['ord_5_0']=test['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)
#test['ord_5_1']=test['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)
#train=train.drop('ord_5',axis=1)
#test=test.drop('ord_5',axis=1)
##comb=rawtrain['ord_5'].unique().tolist()
##comb.remove(np.nan)
##comb=pd.DataFrame({'combine':comb})
##comb['p0']=comb['combine'].apply(lambda x: x[0].upper() if x[0].islower() else x[0].lower())
##comb['p1']=comb['combine'].apply(lambda x: x[1].upper() if x[1].islower() else x[1].lower())
##comb=comb.sort_values(['p0','p1'])
##comb=comb['combine'].tolist()
##combmap={c:i for i,c in enumerate(comb)}
##train['ord_5']=train['ord_5'].replace(combmap)
##test['ord_5']=test['ord_5'].replace(combmap)
##train['ord_5']=train['ord_5_0']+train['ord_5_1']
##test['ord_5']=test['ord_5_0']+test['ord_5_1']
##train=train.drop(['ord_5_0','ord_5_1'],axis=1)
##test=test.drop(['ord_5_0','ord_5_1'],axis=1)
###======encode binary and nominal+label to num for k mode clustering:https://www.kaggle.com/teejmahal20/clustering-categorical-data-k-modes-cat-ii
##normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
##train_cluster=train.drop(normcol59,axis=1)
##test_cluster=test.drop(normcol59,axis=1)
##for c in train_cluster.columns:
##    test_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)
##    train_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)
##
##bincol_labeled=['bin_3', 'bin_4']
##binOE=OrdinalEncoder()
##train_cluster[bincol_labeled]=binOE.fit_transform(train_cluster[bincol_labeled])
##test_cluster[bincol_labeled]=binOE.transform(test_cluster[bincol_labeled])
##
##normcol_labeled=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']
##binOE=OrdinalEncoder()
##train_cluster[normcol_labeled]=binOE.fit_transform(train_cluster[normcol_labeled])
##test_cluster[normcol_labeled]=binOE.transform(test_cluster[normcol_labeled])
##======encode binary+one hot
#bincol=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
#
##======encode nominal
#normcol04=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']
#
##======target encode for nominal
#normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
#
#def FreqEncode(trainc,validc):
#    smap=trainc.value_counts()
#    smap={i:c for i,c in smap.iteritems()}
#    trainc=trainc.map(smap)
#    validc=validc.map(smap)
#    return trainc,validc
#def TargetEncode(trainc,testc,targetc, smooth):
#    print('Target encoding...')
#    smoothing=smooth
#    oof = np.zeros(len(trainc))
#    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(trainc, targetc):
#        train_x=trainc.iloc[tr_idx].reset_index(drop=True)
#        valid_x=trainc.iloc[oof_idx].reset_index(drop=True)
#        target_train=targetc.iloc[tr_idx].reset_index(drop=True)
#        prior = target_train.mean()
#        tmp = target_train.groupby(train_x).agg(['sum', 'count'])
#        tmp['mean'] = tmp['sum'] / tmp['count']
#        smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))
#        cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 
#        tmp['smoothing'] = cust_smoothing
#        tmp = tmp['smoothing'].to_dict()
#        oof[oof_idx]=valid_x.map(tmp).astype(float).fillna(prior).values
#    prior = targetc.mean()
#    tmp = targetc.groupby(trainc).agg(['sum', 'count'])
#    tmp['mean'] = tmp['sum'] / tmp['count']
#    smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))
#    cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 
#    tmp['smoothing'] = cust_smoothing
#    tmp = tmp['smoothing'].to_dict()
#    testc=testc.map(tmp).astype(float).fillna(prior)
#    return oof, testc
#te_list=train.columns.tolist()
#for n in normcol59+normcol04:
#    train[n+'_freq'],test[n+'_freq']=FreqEncode(train[n],test[n])
##    train[n+'_freq'],test[n+'_freq']=train[n+'_freq'].fillna(train[n+'_freq'].mean()),test[n+'_freq'].fillna(train[n+'_freq'].mean())
#for n in normcol59+normcol04+bincol:
#    train[n],test[n]=TargetEncode(train[n],test[n],target,0.3)
#test['nom_6_freq']=test['nom_6_freq'].fillna(train['nom_6_freq'].mean())
    
    
#oof = np.zeros(train[te_list].shape)    
#for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, target):
#        train_x=train.loc[tr_idx,te_list].reset_index(drop=True)
#        valid_x=train.loc[oof_idx,te_list].reset_index(drop=True)
#        target_train=target.iloc[tr_idx].reset_index(drop=True)
#        TE=ce.TargetEncoder(cols=te_list, min_samples_leaf=1, smoothing=0.3)      
#        TE.fit(train_x,target_train)
#        oof[oof_idx,:]=TE.transform(valid_x)
#train.loc[:,te_list]=oof    
#TE=ce.TargetEncoder(cols=te_list, min_samples_leaf=1, smoothing=0.3)      
#TE.fit(train[te_list],target)
#test.loc[:,te_list]=TE.transform(test[te_list])


train = train.sparse.to_coo().tocsr()
test = test.sparse.to_coo().tocsr()
floatlist=['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5_0','ord_5_1']
#for c in floatlist+['day','month']:
#    train[c],test[c]=train[c].fillna(train[c].mean()),test[c].fillna(train[c].mean())
##==================k mode clustering========
#from kmodes.kmodes import KModes
#km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1,random_state=1990)
#train['cluster'] = km.fit_predict(train_cluster)
#test['cluster'] = km.predict(test_cluster)
#==========boost tree MODEL======================
#train=train.drop('ord_5',axis=1)
#test=test.drop('ord_5',axis=1)
#train=train.astype(float)
#test=test.astype(float)
#usedfeatures=test.columns.tolist()#bicolnames+norcolnames+[n+'_miss' for n in normcol59]+[n+'_target' for n in normcol59]+['ord_1','ord_2','ord_3','ord_4','ord_5']
#ss=StandardScaler()
#train[usedfeatures]=ss.fit_transform(train)
#test[usedfeatures]=ss.transform(test)

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=1990)

t1=time.clock()
traintion = np.zeros(train.shape[0])
validation = np.zeros(train.shape[0])
predictions = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train[trn_idx]
    valid_x=train[val_idx]
    target_train=target[trn_idx]
    target_valid=target[val_idx]
    
    LR=LogisticRegression(penalty='l2', tol=0.0001, C=2, fit_intercept=True, intercept_scaling=1, random_state=1990, solver='lbfgs', max_iter=200, verbose=0,  n_jobs=-1)
    LR.fit(train_x,target_train)
    traintion[trn_idx] += LR.predict_proba(train_x)[:,1]/(folds.n_splits-1)
    validation[val_idx] = LR.predict_proba(valid_x)[:,1]
#    
#    fold_importance_df = pd.DataFrame()
#    fold_importance_df["feature"] = usedfeatures
#    fold_importance_df["importance"] = LR.coef_.reshape(-1,1)
#    fold_importance_df["fold"] = fold_ + 1
#    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += LR.predict_proba(test)[:,1] / folds.n_splits
t2=time.clock()-t1
print("Train AUC score: {:<8.5f}".format(roc_auc_score(target,traintion)))
print("Valid AUC score: {:<8.5f}".format(roc_auc_score(target,validation)))
#5cv
#Train AUC score: 0.80053 
#Valid AUC score: 0.78253 
#10cv
#Train AUC score: 0.79882 
#Valid AUC score: 0.78340 
sub['target']=predictions
pd.Series(validation).to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lr_10cv_onehotall_validation.csv',index=False,header=False)
sub.to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lr_10cv_onehotall.csv',index=False)
#============================
train['nom_8'].corr(train['nom_9'])
r5=rawtrain['nom_5'].unique()
rt5=rawtest['nom_5'].unique()
a=set(r5.tolist()+rt5.tolist())
f_noimp_avg = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False))
plt.plot(f_noimp_avg)
usedfeatures=f_noimp_avg.index[f_noimp_avg.importance>2000].tolist()
num_feature=normcol59+['ord_1','ord_2','ord_3','ord_4','ord_5']
cat_cols=[c for c in usedfeatures if c not in num_feature]
