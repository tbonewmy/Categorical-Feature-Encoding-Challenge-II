# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:08:58 2020

@author: wmy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import string

rawtrain=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/train.csv')
rawtest=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/test.csv')
sub=pd.read_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sample_submission.csv')
target=rawtrain['target']
rawtrain.columns
rawtrain.nunique()
#a=train.head(10)
target.value_counts()
string.ascii_letters
train=rawtrain.drop(['id','target'],axis=1)
test=rawtest.drop('id',axis=1)
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
lowletter=rawtrain['ord_3'].unique().tolist()
lowletter.remove(np.nan)
lowletter.sort()
lowermap={c:i for i,c in enumerate(lowletter)}
train['ord_3']=train['ord_3'].replace(lowermap)
test['ord_3']=test['ord_3'].replace(lowermap)
uppermap={c:i for i,c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
train['ord_4']=train['ord_4'].replace(uppermap)
test['ord_4']=test['ord_4'].replace(uppermap)
comb=rawtrain['ord_5'].unique().tolist()
comb.remove(np.nan)
comb=pd.DataFrame({'combine':comb})
comb['p0']=comb['combine'].apply(lambda x: x[0].upper() if x[0].islower() else x[0].lower())
comb['p1']=comb['combine'].apply(lambda x: x[1].upper() if x[1].islower() else x[1].lower())
comb=comb.sort_values(['p0','p1'])
comb=comb['combine'].tolist()
combmap={c:i for i,c in enumerate(comb)}
train['ord_5']=train['ord_5'].replace(combmap)
test['ord_5']=test['ord_5'].replace(combmap)
#======encode binary
bincol=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
binOH=OneHotEncoder(sparse=False)
train[bincol]=train[bincol].astype(str).fillna('Na')
test[bincol]=test[bincol].astype(str).fillna('Na')
bindata=binOH.fit_transform(train[bincol])
bindatatest=binOH.transform(test[bincol])
bicolnames=[b+'_'+a for i,b in enumerate(bincol) for a in binOH.categories_[i]]
for c in bicolnames:
    train[c]=0
    test[c]=0
train[bicolnames]=bindata
test[bicolnames]=bindatatest
train=train.drop(bincol,axis=1)
test=test.drop(bincol,axis=1)
#======encode nominal
normcol04=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']
train[normcol04].nunique()
norm04OH=OneHotEncoder(sparse=False)
train[normcol04]=train[normcol04].astype(str).fillna('Na')
test[normcol04]=test[normcol04].astype(str).fillna('Na')
norm04data=norm04OH.fit_transform(train[normcol04])
norm04datatest=norm04OH.transform(test[normcol04])
norcolnames=[b+'_'+a for i,b in enumerate(normcol04) for a in norm04OH.categories_[i]]
for c in norcolnames:
    train[c]=0
    test[c]=0
train[norcolnames]=norm04data
test[norcolnames]=norm04datatest
train=train.drop(normcol04,axis=1)
test=test.drop(normcol04,axis=1)
# 
normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
def SupervisedRatio(trainc,validc,targetc):
    block=pd.DataFrame({'class':trainc,'target':targetc})
    smap=block.groupby('class')['target'].mean()
    smap={c:v for c,v in smap.iteritems()}
    trainc=trainc.map(smap)
    validc=validc.map(smap)
    return trainc,validc
for n in normcol59:
        _,test[n]=SupervisedRatio(train[n],test[n],target)
   
#==========boost tree MODEL======================
usedfeatures=train.columns.tolist()
cat_cols=bicolnames+norcolnames
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=4590)
#params = {#test
##    'bagging_freq': 5,
#    'max_bin':30,
##    'bagging_fraction': 0.8,
##    'boost_from_average':'false',
#    'boost': 'gbdt',
#    'feature_fraction': 0.6,
#    'learning_rate': 0.01,
#    'max_depth': 8,
#    'min_data_in_leaf': 10,
##    'min_sum_hessian_in_leaf': 10.0,
#    'num_leaves': 10,
#    'num_threads': 8,
##    'tree_learner': 'serial',
##        "lambda_l1" : 0.5,
##    "lambda_l2" : 0.2,
#    "metric" : "rmse",
#    'objective': 'regression',
#    'verbosity': 1}
params = {#test
#    'bagging_freq': 5,
    'max_bin':30,
#    'bagging_fraction': 0.8,
#    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.6,
    'learning_rate': 0.1,
    'max_depth': 8,
    'min_data_in_leaf': 10,
#    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'num_threads': 8,
#    'tree_learner': 'serial',
#        "lambda_l1" : 0.5,
#    "lambda_l2" : 0.2,
    "metric" : "auc",
    'objective': 'binary',
    'verbosity': 1}
traintion = np.zeros(len(train))
validation = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train.iloc[trn_idx].reset_index(drop=True)
    valid_x=train.iloc[val_idx].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)
    for n in normcol59:
        train_x[n],valid_x[n]=SupervisedRatio(train_x[n],valid_x[n],target_train)
    train_x=train_x.astype(float)
    valid_x=valid_x.astype(float)
    trn_data = lgb.Dataset(train_x,
                           label=target_train,
                           categorical_feature=cat_cols
                          )
    val_data = lgb.Dataset(valid_x,
                           label=target_valid,
                           categorical_feature=cat_cols
                          )

    num_round = 1000000
    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=250,
                    early_stopping_rounds = 500)
    traintion[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)
    validation[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = usedfeatures
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
print("Train AUC score: {:<8.5f}".format(roc_auc_score(target,traintion)))
print("Valid AUC score: {:<8.5f}".format(roc_auc_score(target,validation)))
#Train AUC score: 0.80428 
#Valid AUC score: 0.78394 
sub['target']=predictions
sub.to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/subbench.csv',index=False)
