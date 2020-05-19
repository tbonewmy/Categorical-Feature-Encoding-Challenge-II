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
from sklearn.naive_bayes import CategoricalNB
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
bincol=['bin_3', 'bin_4']
normcol04=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']
normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for c in bincol+normcol04+normcol59:
    levels=train[c].unique().tolist()
    levels.remove(np.nan)
    tempmap={e:i for i,e in enumerate(levels)}
    train[c]=train[c].map(tempmap)
    test[c]=test[c].map(tempmap)
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
train=train.fillna(999)
test=test.fillna(999)
usedfeatures=test.columns.tolist()#bicolnames+norcolnames+[n+'_miss' for n in normcol59]+[n+'_target' for n in normcol59]+['ord_1','ord_2','ord_3','ord_4','ord_5']
# cat_cols=['cluster']#+[n+'_freq' for n in normcol59]#+['ord_1','ord_2','ord_3','ord_4','ord_5']+[n+'_miss' for n in normcol59]+bicolnames+norcolnames+
folds = StratifiedKFold(n_splits=40, shuffle=True, random_state=1990)

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


    CB=CategoricalNB(alpha=5)
    CB.fit(train_x,target_train)
    traintion[trn_idx] += CB.predict_proba(train_x)[:,1]/(folds.n_splits-1)
    validation[val_idx] = CB.predict_proba(valid_x)[:,1]
    

    predictions += CB.predict_proba(test[usedfeatures])[:,1] / folds.n_splits
t2=time.clock()-t1
print("Train AUC score: {:<8.5f}".format(roc_auc_score(target,traintion)))
print("Valid AUC score: {:<8.5f}".format(roc_auc_score(target,validation)))
#0.5
# Train AUC score: 0.79510 
# Valid AUC score: 0.78065 
# 1
# Train AUC score: 0.79500 
# Valid AUC score: 0.78083 
# 5
# Train AUC score: 0.79385 
# Valid AUC score: 0.78095 
# 10
# Train AUC score: 0.79217 
# Valid AUC score: 0.78026 
# 20cv
# Train AUC score: 0.79332 
# Valid AUC score: 0.78125 
# 40cv
# Train AUC score: 0.79306 
# Valid AUC score: 0.78145 
sub['target']=predictions
pd.Series(validation).to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/cb_40cv_labelencode_cluster_validation.csv',index=False,header=False)
sub.to_csv('E:/Kaggle/Categorical Feature Encoding Challenge II/sub/cb_40cv_labelencode_cluster.csv',index=False)
