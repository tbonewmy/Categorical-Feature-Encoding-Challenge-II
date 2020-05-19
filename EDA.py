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
from sklearn.linear_model import LinearRegression
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
#rawtrain.isna().sum()
#rawtrain.columns
#rawtrain.nunique()
#a=train.head(10)
#target.value_counts()
#string.ascii_letters
train=rawtrain.drop(['id','target'],axis=1)
test=rawtest.drop('id',axis=1)
#rawtrain['month'].plot(kind='bar')
#sns.catplot('target', col="ord_2",col_wrap=4, data=rawtrain,
#                height=2.5, kind="count")

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
#comb=rawtrain['ord_5'].unique().tolist()
#comb.remove(np.nan)
#comb=pd.DataFrame({'combine':comb})
#comb['p0']=comb['combine'].apply(lambda x: x[0].upper() if x[0].islower() else x[0].lower())
#comb['p1']=comb['combine'].apply(lambda x: x[1].upper() if x[1].islower() else x[1].lower())
#comb=comb.sort_values(['p0','p1'])
#comb=comb['combine'].tolist()
#combmap={c:i for i,c in enumerate(comb)}
#train['ord_5']=train['ord_5'].replace(combmap)
#test['ord_5']=test['ord_5'].replace(combmap)
#train['ord_5']=train['ord_5_0']+train['ord_5_1']
#test['ord_5']=test['ord_5_0']+test['ord_5_1']
#train=train.drop(['ord_5_0','ord_5_1'],axis=1)
#test=test.drop(['ord_5_0','ord_5_1'],axis=1)
#======encode binary and nominal+label to num for k mode clustering:https://www.kaggle.com/teejmahal20/clustering-categorical-data-k-modes-cat-ii
normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
train_cluster=train.drop(normcol59,axis=1)
test_cluster=test.drop(normcol59,axis=1)
for c in train_cluster.columns:
    test_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)
    train_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)

bincol_labeled=['bin_3', 'bin_4']
binOE=OrdinalEncoder()
train_cluster[bincol_labeled]=binOE.fit_transform(train_cluster[bincol_labeled])
test_cluster[bincol_labeled]=binOE.transform(test_cluster[bincol_labeled])

normcol_labeled=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']
binOE=OrdinalEncoder()
train_cluster[normcol_labeled]=binOE.fit_transform(train_cluster[normcol_labeled])
test_cluster[normcol_labeled]=binOE.transform(test_cluster[normcol_labeled])
#======encode binary+one hot
bincol=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
#binOH=OneHotEncoder(sparse=False)
#train[bincol]=train[bincol].astype(str).fillna('Na')
#test[bincol]=test[bincol].astype(str).fillna('Na')
#bindata=binOH.fit_transform(train[bincol])
#bindatatest=binOH.transform(test[bincol])
#bicolnames=[b+'_'+a for i,b in enumerate(bincol) for a in binOH.categories_[i]]
#for c in bicolnames:
#    train[c]=0
#    test[c]=0
#train[bicolnames]=bindata
#test[bicolnames]=bindatatest
#train=train.drop(bincol,axis=1)
#test=test.drop(bincol,axis=1)

#======encode nominal
normcol04=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']
#train[normcol04].nunique()
#norm04OH=OneHotEncoder(sparse=False)
#train[normcol04]=train[normcol04].astype(str).fillna('Na')
#test[normcol04]=test[normcol04].astype(str).fillna('Na')
#norm04data=norm04OH.fit_transform(train[normcol04])
#norm04datatest=norm04OH.transform(test[normcol04])
#norcolnames=[b+'_'+a for i,b in enumerate(normcol04) for a in norm04OH.categories_[i]]
#for c in norcolnames:
#    train[c]=0
#    test[c]=0
#train[norcolnames]=norm04data
#test[norcolnames]=norm04datatest
#train=train.drop(normcol04,axis=1)
#test=test.drop(normcol04,axis=1)
#======target encode for nominal
normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
#def TargetEncode(trainc,validc,targetc):
##    te=ce.TargetEncoder(smoothing=0.3)
##    te.fit(trainc.values,targetc.values)
##    validc=te.transform(validc.values)
##    trainc=te.transform(trainc.values)
#    block=pd.DataFrame({'class':trainc,'target':targetc})
#    smap=block.groupby('class')['target'].mean()
#    smap={c:v for c,v in smap.iteritems()}
#    trainc=trainc.map(smap)
##    filler=trainc.mean()
##    trainc.fillna(filler,inplace=True)
#    validc=validc.map(smap)
##    validc.fillna(filler,inplace=True)
#    return trainc,validc
def FreqEncode(trainc,validc):
    smap=trainc.value_counts()
    smap={i:c for i,c in smap.iteritems()}
    trainc=trainc.map(smap)
    validc=validc.map(smap)
    return trainc,validc
#monthday=['day','month']
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
        oof[oof_idx]=valid_x.map(tmp).values
    prior = targetc.mean()
    tmp = targetc.groupby(trainc).agg(['sum', 'count'])
    tmp['mean'] = tmp['sum'] / tmp['count']
    smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))
    cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 
    tmp['smoothing'] = cust_smoothing
    tmp = tmp['smoothing'].to_dict()
    testc=testc.map(tmp)
    return oof, testc
for n in normcol59+normcol04:
    train[n+'_freq'],test[n+'_freq']=FreqEncode(train[n],test[n])
for n in normcol59+normcol04+bincol:
    train[n],test[n]=TargetEncode(train[n],test[n],target,0.3)
#    train[n+'_miss']=train[n].isna()
#    test[n+'_miss']=test[n].isna()
train['total_miss']=train.isna().sum(axis=1)
test['total_miss']=test.isna().sum(axis=1)
#train['total_miss_square']=train['total_miss']**2
#test['total_miss_square']=test['total_miss']**2
#for n in bincol+normcol04+normcol59+monthday:
##    _,test[n+'_freq']=FreqEncode(train[n],test[n])
#    _,test[n+'_target']=TargetEncode(train[n],test[n],target)
##    train[n+'_miss']=train[n].isna()
##    test[n+'_miss']=test[n].isna()
#te=ce.TargetEncoder(smoothing=0.3)
#te.fit(train,target)
#test=te.transform(test)
#==================k mode clustering========
from kmodes.kmodes import KModes
km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1,random_state=1990)
train['cluster'] = km.fit_predict(train_cluster)
test['cluster'] = km.predict(test_cluster)
##==========test independency
#import scipy.stats as scs
#
#def chi_square_of_df_cols(df, col1, col2):
#    df_col1, df_col2 = df[col1], df[col2]
#
#    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
#               for cat2 in df_col2.unique()]
#              for cat1 in df_col1.unique()]
#
#    return scs.chi2_contingency(result)
#
#chi_matrix=np.zeros([len(train_cluster.columns),len(train_cluster.columns)])
#for i,r in enumerate(train_cluster.columns):
#    for j,c in enumerate(train_cluster.columns):
#        print('{}{}'.format(i,j),flush=True)
#        if i!=j:
#            stemp,_,_,_=chi_square_of_df_cols(train_cluster, r, c)
#            chi_matrix[i,j]=stemp/(train_cluster[r].nunique()*train_cluster[c].nunique())
#
#
## Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(12, 12))
#
#sns.heatmap(pd.DataFrame(chi_matrix,columns=train_cluster.columns,index=train_cluster.columns), 
#             cmap="Greens", annot=True, square=True, linewidths=.5)
#======================================
#for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, target):
#        ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
#        ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])
#        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
#    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
#    ce_target_encoder.fit(train, target)
#    train = oof.sort_index()
#    test = ce_target_encoder.transform(test)
#============month to days
#monthmap={i+1:c for i,c in enumerate([31,28,31,30,31,30,31,31,30,31,30,31])}
#def month2days(x):
#    if np.isnan(x):
#        return x
#    elif x==1:
#        return 0
#    else:
#        days=0
#        for i in range(int(x-1)):
#            days += monthmap[i+1]
#        return days
#train['days']=train['month'].apply(lambda x: month2days(x))+train['day']
#test['days']=test['month'].apply(lambda x: month2days(x))+test['day']
#==========boost tree MODEL======================
#norcolnames bincol+
#train=train.drop('ord_5',axis=1)
#test=test.drop('ord_5',axis=1)
floatlist=['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5_0','ord_5_1']
train[floatlist]=train[floatlist].astype(float)
test[floatlist]=test[floatlist].astype(float)
usedfeatures=test.columns.tolist()#bicolnames+norcolnames+[n+'_miss' for n in normcol59]+[n+'_target' for n in normcol59]+['ord_1','ord_2','ord_3','ord_4','ord_5']
cat_cols=['cluster']#+[n+'_freq' for n in normcol59]#+['ord_1','ord_2','ord_3','ord_4','ord_5']+[n+'_miss' for n in normcol59]+bicolnames+norcolnames+
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1990)
#params = {#test
#        'learning_rate': 0.05,
##        'feature_fraction': 0.1,
#        'min_data_in_leaf' : 100,
#        'max_depth': 5,
#        'reg_alpha': 30,#l1
##        'reg_lambda': 1,#l2
#        'objective': 'binary',
##        'num_leaves':30,
#        'metric': 'auc',
#        'n_jobs': -1,
##        'feature_fraction_seed': 42,
#        'bagging_seed': 42,
#        'boosting_type': 'gbdt',
#        'verbose': 1,
#        'is_unbalance': True,
#        'bagging_freq':5,
#        'pos_bagging_fraction':0.8,
#        'neg_bagging_fraction':0.8,
#        'boost_from_average': False
#        }
params = {#1
        'learning_rate': 0.05,
        'feature_fraction': 0.1,
        'min_data_in_leaf' : 50,
        'max_depth': 5,
#        'max_bin':300,
#        'reg_alpha': 10,#l1
#        'reg_lambda': 30,#l2
        'objective': 'binary',
        'num_leaves':3,
        'metric': 'auc',
        'n_jobs': -1,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'boosting_type': 'gbdt',
        'verbose': 1,
        'is_unbalance': True,
#        'bagging_freq':5,
#        'pos_bagging_fraction':0.8,
#        'neg_bagging_fraction':0.8,
        'boost_from_average': False}
t1=time.clock()
traintion = np.zeros(len(train))
validation = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold n°{}".format(fold_))
    train_x=train.iloc[trn_idx][usedfeatures].reset_index(drop=True)
    valid_x=train.iloc[val_idx][usedfeatures].reset_index(drop=True)
    target_train=target.iloc[trn_idx].reset_index(drop=True)
    target_valid=target.iloc[val_idx].reset_index(drop=True)
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
                    early_stopping_rounds = 250)
    traintion[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)
    validation[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = usedfeatures
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[usedfeatures], num_iteration=clf.best_iteration) / folds.n_splits
t2=time.clock()-t1
print("Train AUC score: {:<8.5f}".format(roc_auc_score(target,traintion)))
print("Valid AUC score: {:<8.5f}".format(roc_auc_score(target,validation)))
#Train AUC score: 0.79086 
#Valid AUC score: 0.78817 
#2ord5
#Train AUC score: 0.79038 
#Valid AUC score: 0.78821 
#2ord5+no_freq
#Train AUC score: 0.79020 
#Valid AUC score: 0.78828 
#2ord5+freq+20cv
#Train AUC score: 0.79030 
#Valid AUC score: 0.78825 
#2ord5+freq+10cv+depth=5+leaves=3
#Train AUC score: 0.79038 
#Valid AUC score: 0.78821 
#2ord5+freq+10cv+depth=5+leaves=3+10cv_targetencode
#Train AUC score: 0.79106 
#Valid AUC score: 0.78869 
#2ord5+freq+10cv+depth=5+leaves=3+10cv_targetencode
# Train AUC score: 0.79093 
# Valid AUC score: 0.78864 
sub['target']=predictions
pd.Series(validation).to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lgb_10cv_param_LeOd56789_FeqNo_10cvTeNoBi_2ord5_allmiss_cluster_validation.csv',index=False,header=False)
sub.to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lgb_10cv_param_LeOd56789_FeqNo_10cvTeNoBi_2ord5_allmiss_cluster.csv',index=False)
pd.DataFrame.from_dict(params,orient='index').to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/lgb_10cv_param_LeOd56789_FeqNo_10cvTeNoBi_2ord5_allmiss_cluster_params.csv',index=False)
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
#Train AUC score: 0.79045 
#Valid AUC score: 0.78817 

#Train AUC score: 0.79070 
#Valid AUC score: 0.78833 

#Train AUC score: 0.79074 
#Valid AUC score: 0.78835 
#sub_10cv_param_LeOd56789_TeNoBi_allmiss_cluster
#Train AUC score: 0.79049 
#Valid AUC score: 0.78833 
#sub_10cv_paramcat_LeOd56789_TeNo_allmiss_cluster
#Train AUC score: 0.80368 
#Valid AUC score: 0.78605 
#sub_10cv_param_LeOd56789_FeqNoBi_TeNoBi_allmiss_cluster
#Train AUC score: 0.80456 
#Valid AUC score: 0.78607 
#sub_10cv_param_LeOd56789_FeqNo_TeNoBi_allmiss_cluster
#Train AUC score: 0.79073 
#Valid AUC score: 0.78829 
#sub_10cv_param_LeOd56789_FeqNo_TeNoBi_allmiss_cluster:min_sample_leaf=12-100
#Train AUC score: 0.79058 
#Valid AUC score: 0.78830
#sub_20cv_param_LeOd56789_FeqNo_TeNoBi_allmiss_cluster:min_sample_leaf=12-100
#Train AUC score: 0.79032 
#Valid AUC score: 0.78833 