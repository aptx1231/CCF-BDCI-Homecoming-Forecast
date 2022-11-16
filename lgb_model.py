import nltk
import re
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, classification_report
from urllib.parse import unquote, urlparse, quote
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

# 数据读取与拼接
train = pd.read_csv('data/dataTrain.csv')
test = pd.read_csv('data/dataA.csv')
data = pd.concat([train, test]).reset_index(drop=True)
data['f3'] = data['f3'].map({'low': 0, 'mid': 1, 'high': 2})
print(data.shape)

enable_log = True

enable_10p = True
enable_10d = False
enable_10a = False
enable_10s = False
enable_105 = False
enable_4 = True

enable_loc_com = True

enable_10p_com = True
enable_10d_com = False
enable_10a_com = False
enable_10s_com = False
enable_105_com = False
enable_4_com = True

enable_10p_inter = True
enable_10d_inter = False
enable_10a_inter = False
enable_10s_inter = False
enable_105_inter = False
enable_4_inter = False

print(enable_10p, enable_10d, enable_105, enable_4, enable_10a, enable_10s)
print(enable_10p_com, enable_10d_com, enable_105_com, enable_4_com, enable_10a_com, enable_10s_com)
print(enable_10p_inter, enable_10d_inter, enable_105_inter, enable_4_inter, enable_10a_inter, enable_10s_inter)
print(enable_loc_com, enable_log)

kfold = 10
name = 'res/FX_sub_lgb-{}-select50000-origin-10plusall-comlocjh.csv'.format(kfold)
print(name)

# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
if enable_4:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            data[f'{loc_f[i]}+{loc_f[j]}'] = data[loc_f[i]] + data[loc_f[j]]
            data[f'{loc_f[i]}-{loc_f[j]}'] = data[loc_f[i]] - data[loc_f[j]]
            data[f'{loc_f[i]}*{loc_f[j]}'] = data[loc_f[i]] * data[loc_f[j]]
            data[f'{loc_f[i]}/{loc_f[j]}'] = data[loc_f[i]] / data[loc_f[j]]
if enable_10p or enable_10d:
    for i in range(len(loc_f)):
        for j in range(len(loc_f)):
            if i == j:
                continue
            if enable_10p:
                data[f'10p{loc_f[i]}+{loc_f[j]}'] = data[loc_f[i]] * 10 + data[loc_f[j]]
            if enable_10d:
                data[f'10d{loc_f[i]}+{loc_f[j]}'] = data[loc_f[i]] / 10 + data[loc_f[j]]
if enable_105:
    for i in range(len(loc_f)):
        for j in range(len(loc_f)):
            if i == j:
                continue
            for k in range(len(loc_f)):
                if k == j or k == i:
                    continue
                # print(loc_f[i], loc_f[j], loc_f[k])
                data[f'105p{loc_f[i]}+{loc_f[j]}+{loc_f[k]}'] = data[loc_f[i]] * 10 + data[loc_f[j]] * 5 + data[loc_f[j]]

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
if enable_4_com:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            data[f'{com_f[i]}+{com_f[j]}'] = data[com_f[i]] + data[com_f[j]]
            data[f'{com_f[i]}-{com_f[j]}'] = data[com_f[i]] - data[com_f[j]]
            data[f'{com_f[i]}*{com_f[j]}'] = data[com_f[i]] * data[com_f[j]]
            data[f'{com_f[i]}/{com_f[j]}'] = data[com_f[i]] / data[com_f[j]]
if enable_10p_com or enable_10d_com:
    for i in range(len(com_f)):
        for j in range(len(com_f)):
            if i == j:
                continue
            if enable_10p_com:
                data[f'10p{com_f[i]}+{com_f[j]}'] = data[com_f[i]] * 10 + data[com_f[j]]
            if enable_10d_com:
                data[f'10d{com_f[i]}+{com_f[j]}'] = data[com_f[i]] / 10 + data[com_f[j]]
if enable_105_com:
    for i in range(len(com_f)):
        for j in range(len(com_f)):
            if i == j:
                continue
            for k in range(len(com_f)):
                if k == j or k == i:
                    continue
                # print(com_f[i], com_f[j], com_f[k])
                data[f'105p{com_f[i]}+{com_f[j]}+{com_f[k]}'] = data[com_f[i]] * 10 + data[com_f[j]] * 5 + data[com_f[j]]

if enable_loc_com:
    for i in range(len(com_f)):
        for j in range(len(loc_f)):
            if i == j:
                continue
            data[f'10p{com_f[i]}+{loc_f[j]}'] = data[com_f[i]] * 10 + data[loc_f[j]]
    for i in range(len(loc_f)):
        for j in range(len(com_f)):
            if i == j:
                continue
            data[f'10p{loc_f[i]}+{com_f[j]}'] = data[loc_f[i]] * 10 + data[com_f[j]]

if enable_log:
    # 离散化
    all_f = [f'f{idx}' for idx in range(1, 47) if idx != 3]
    for col in all_f:
        data[f'{col}_log'] = data[col].apply(lambda x: int(np.log(x)) if x > 0 else 0)

    # 特征交叉
    log_f = [f'f{idx}_log' for idx in range(1, 47) if idx != 3]
    for i in range(len(log_f)):
        for j in range(i + 1, len(log_f)):
            data[f'{log_f[i]}_{log_f[j]}'] = data[log_f[i]]*10000 + data[log_f[j]]


# # 互联网特征
int_f = ['f3','f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17',
       'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26',
       'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35',
       'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42']
if enable_10p_inter or enable_10d_inter:
    for i in range(len(int_f)):
        for j in range(len(int_f)):
            if i == j:
                continue
            if enable_10p_inter:
                data[f'10p{int_f[i]}+{int_f[j]}'] = data[int_f[i]] * 10 + data[int_f[j]]
            if enable_10d_inter:
                data[f'10d{int_f[i]}+{int_f[j]}'] = data[int_f[i]] / 10 + data[int_f[j]]
if enable_105_inter:
    for i in range(len(int_f)):
        for j in range(len(int_f)):
            if i == j:
                continue
            for k in range(len(int_f)):
                if k == j or k == i:
                    continue
                # print(com_f[i], com_f[j], com_f[k])
                data[f'105p{int_f[i]}+{int_f[j]}+{int_f[k]}'] = data[int_f[i]] * 10 + data[int_f[j]] * 5 + data[int_f[j]]
if enable_4_inter:
    for i in range(len(int_f)):
        for j in range(i + 1, len(int_f)):
            data[f'{int_f[i]}+{int_f[j]}'] = data[int_f[i]] + data[int_f[j]]
            data[f'{int_f[i]}-{int_f[j]}'] = data[int_f[i]] - data[int_f[j]]
            data[f'{int_f[i]}*{int_f[j]}'] = data[int_f[i]] * data[int_f[j]]
            data[f'{int_f[i]}/{int_f[j]}'] = data[int_f[i]] / data[int_f[j]]

# features = [i for i in train.columns if i not in ['label', 'id']]  # 原始特征名
# other_features = [i for i in data.columns if i not in features]   # 扩展特征名
# print(len(features), len(other_features))
#
# X = data[features]
# poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
# X_ploly = poly.fit_transform(X)
# X_ploly_df = pd.DataFrame(X_ploly, columns=poly.get_feature_names())
# print(X_ploly_df.shape)
#
# data = pd.concat([X_ploly_df, data[other_features]], axis=1)
print(data.shape)
# import sys
# sys.exit()

# 训练测试分离
train = data[~data['label'].isna()].reset_index(drop=True)[:50000]
test = data[data['label'].isna()].reset_index(drop=True)

features = [i for i in train.columns if i not in ['label', 'id']]
print(len(features))
y = train['label']

KF = StratifiedKFold(n_splits=kfold, random_state=2021, shuffle=True)
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': 30,
    'learning_rate': 0.05,
    'num_leaves': 2 ** 6,
    'max_depth': 8,
    'tree_learner': 'serial',
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,
    'max_bin': 255,
    'verbose': -1,
    'seed': 2021,
    'bagging_seed': 2021,
    'feature_fraction_seed': 2021,
    'early_stopping_rounds': 100,
}

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros((len(test)))

# 模型训练
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 3000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=50,
    )

    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(test[features], num_iteration=clf.best_iteration) / kfold
    feat_imp_df['imp'] += clf.feature_importance() / kfold

print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

# 提交结果
test['label'] = predictions_lgb
print(name)
test[['id', 'label']].to_csv(name, index=False)

