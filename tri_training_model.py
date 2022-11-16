import nltk
import re
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
import scipy.sparse as sp

import sklearn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--kind', type=str, nargs='+')
args = parser.parse_args()

train_data = pd.read_csv('data/dataTrain.csv')  # train_df (59872, 48)
test_data = pd.read_csv('data/dataA.csv')
submission = pd.read_csv('data/submit_example_A.csv')
data_nolabel = pd.read_csv('data/dataNoLabel.csv')
print(f'origin train_data.shape = {train_data.shape}')
print(f'origin test_data.shape  = {test_data.shape}')
print(f'origin data_nolabel.shape  = {data_nolabel.shape}')
print(args.kind[0], args.kind[1], args.kind[2])

enable_log = True

enable_10p = False
enable_10d = False
enable_10a = False
enable_10s = False
enable_105 = False
enable_4 = True

enable_loc_com = False

enable_10p_com = False
enable_10d_com = False
enable_10a_com = False
enable_10s_com = False
enable_105_com = False
enable_4_com = True

enable_10p_inter = False
enable_10d_inter = False
enable_10a_inter = False
enable_10s_inter = False
enable_105_inter = False
enable_4_inter = False

kfold = 10
gpu_id = 0
kind = 'tri-{}-{}-{}'.format(args.kind[0], args.kind[1], args.kind[2])
name = 'res/FX_sub_B-{}-{}-lgbmodel3-log4.csv'.format(kind, kfold)

print(kfold, name)
print(enable_10p, enable_10d, enable_105, enable_4, enable_10a, enable_10s)
print(enable_10p_com, enable_10d_com, enable_105_com, enable_4_com, enable_10a_com, enable_10s_com)
print(enable_10p_inter, enable_10d_inter, enable_105_inter, enable_4_inter, enable_10a_inter, enable_10s_inter)
print(enable_loc_com)

# 离散化f3
train_data['f3'] = train_data['f3'].map({'low': 0, 'mid': 1, 'high': 2})
test_data['f3'] = test_data['f3'].map({'low': 0, 'mid': 1, 'high': 2})
data_nolabel['f3'] = data_nolabel['f3'].map({'low': 0, 'mid': 1, 'high': 2})

# 暴力Feature
if enable_4:
    loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(loc_f)):
            for j in range(i + 1, len(loc_f)):
                df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
                df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
                df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
                df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)
if enable_10p or enable_10d:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(loc_f)):
            for j in range(len(loc_f)):
                if i == j:
                    continue
                if enable_10p:
                    df[f'10p{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] * 10 + df[loc_f[j]]
                if enable_10d:
                    df[f'10d{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] / 10 + df[loc_f[j]]
if enable_105:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(loc_f)):
            for j in range(len(loc_f)):
                if i == j:
                    continue
                for k in range(len(loc_f)):
                    if k == j or k == i:
                        continue
                    # print(loc_f[i], loc_f[j], loc_f[k])
                    df[f'105p{loc_f[i]}+{loc_f[j]}+{loc_f[k]}'] = df[loc_f[i]] * 10 + df[loc_f[j]] * 5 + df[loc_f[j]]

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
if enable_4_com:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(com_f)):
            for j in range(i + 1, len(com_f)):
                df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
                df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
                df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
                df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)
if enable_10p_com or enable_10d_com:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(com_f)):
            for j in range(len(com_f)):
                if i == j:
                    continue
                if enable_10p_com:
                    df[f'10p{com_f[i]}+{com_f[j]}'] = df[com_f[i]] * 10 + df[com_f[j]]
                if enable_10d_com:
                    df[f'10d{com_f[i]}+{com_f[j]}'] = df[com_f[i]] / 10 + df[com_f[j]]
if enable_105_com:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(com_f)):
            for j in range(len(com_f)):
                if i == j:
                    continue
                for k in range(len(com_f)):
                    if k == j or k == i:
                        continue
                    # print(com_f[i], com_f[j], com_f[k])
                    df[f'105p{com_f[i]}+{com_f[j]}+{com_f[k]}'] = df[com_f[i]] * 10 + df[com_f[j]] * 5 + df[com_f[j]]

if enable_loc_com:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(com_f)):
            for j in range(len(loc_f)):
                if i == j:
                    continue
                df[f'10p{com_f[i]}+{loc_f[j]}'] = df[com_f[i]] * 10 + df[loc_f[j]]
        for i in range(len(loc_f)):
            for j in range(len(com_f)):
                if i == j:
                    continue
                df[f'10p{loc_f[i]}+{com_f[j]}'] = df[loc_f[i]] * 10 + df[com_f[j]]

if enable_log:
    # 离散化
    all_f = [f'f{idx}' for idx in range(1, 47) if idx != 3]
    for df in [train_data, test_data, data_nolabel]:
        for col in all_f:
            df[f'{col}_log'] = df[col].apply(lambda x: int(np.log(x)) if x > 0 else 0)

    # 特征交叉
    log_f = [f'f{idx}_log' for idx in range(1, 47) if idx != 3]
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(log_f)):
            for j in range(i + 1, len(log_f)):
                df[f'{log_f[i]}_{log_f[j]}'] = df[log_f[i]]*10000 + df[log_f[j]]

# # 互联网特征
int_f = ['f3','f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17',
       'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26',
       'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35',
       'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42']
if enable_10p_inter or enable_10d_inter:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(int_f)):
            for j in range(len(int_f)):
                if i == j:
                    continue
                if enable_10p_inter:
                    df[f'10p{int_f[i]}+{int_f[j]}'] = df[int_f[i]] * 10 + df[int_f[j]]
                if enable_10d_inter:
                    df[f'10d{int_f[i]}+{int_f[j]}'] = df[int_f[i]] / 10 + df[int_f[j]]
if enable_105_inter:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(int_f)):
            for j in range(len(int_f)):
                if i == j:
                    continue
                for k in range(len(int_f)):
                    if k == j or k == i:
                        continue
                    # print(com_f[i], com_f[j], com_f[k])
                    df[f'105p{int_f[i]}+{int_f[j]}+{int_f[k]}'] = df[int_f[i]] * 10 + df[int_f[j]] * 5 + df[int_f[j]]
if enable_4_inter:
    for df in [train_data, test_data, data_nolabel]:
        for i in range(len(int_f)):
            for j in range(i + 1, len(int_f)):
                df[f'{int_f[i]}+{int_f[j]}'] = df[int_f[i]] + df[int_f[j]]
                df[f'{int_f[i]}-{int_f[j]}'] = df[int_f[i]] - df[int_f[j]]
                df[f'{int_f[i]}*{int_f[j]}'] = df[int_f[i]] * df[int_f[j]]
                df[f'{int_f[i]}/{int_f[j]}'] = df[int_f[i]] / df[int_f[j]]

print(f'processed train_data.shape = {train_data.shape}')
print(f'processed test_data.shape  = {test_data.shape}')
print(f'processed data_nolabel.shape  = {data_nolabel.shape}')

feature_columns = [ col for col in train_data.columns if col not in ['id', 'label']]
print(len(feature_columns))
target = 'label'

# print(feature_columns)
train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]
nolabel = data_nolabel[feature_columns]
print(train.shape, label.shape, test.shape, nolabel.shape)
print(type(train), type(label), type(test))

train = train[:50000]
label = label[:50000]
print(train.shape, label.shape, test.shape, nolabel.shape)
print(type(train), type(label), type(test))


class Tri_Training():
    def __init__(self, base_estimator, base_estimator_2, base_estimator_3, verbose=500):
        self.estimators=[base_estimator,base_estimator_2,base_estimator_3]
        self.verbose = verbose

    def fit(self, X, y, X_val, y_val, unlabeled_X):
        self.iter = 0
        print('='*10, 'iter', self.iter, '='*10)
        for i in range(3):
            print('### training estimator', i)
            sample = sklearn.utils.resample(X, y) # bootstrap 重采样
            self.estimators[i].fit(*sample, eval_set=[(X_val, y_val)], verbose=self.verbose)
        e_prime = [0.5] * 3 # t-1轮分类器的准确率
        l_prime = [0] * 3 # t-1轮被标注的样本数
        e = [0] * 3
        update = [False] * 3
        lb_X, lb_y = [[]] * 3, [[]] * 3
        improve = True
        while improve:
            self.iter += 1
            print('='*10, 'iter', self.iter, '='*10)
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(X, y, j, k)
                if e[i] < e_prime[i]:
                    # ulb_y_j = self.estimators[j].predict(unlabeled_X, num_iteration=self.estimators[j].best_iteration_)
                    # ulb_y_k = self.estimators[k].predict(unlabeled_X, num_iteration=self.estimators[k].best_iteration_)
                    ulb_y_j = self.estimators[j].predict(unlabeled_X)
                    ulb_y_k = self.estimators[k].predict(unlabeled_X)
                    lb_X[i] = unlabeled_X[ulb_y_j == ulb_y_k]
                    lb_y[i] = ulb_y_j[ulb_y_j == ulb_y_k]
                    if l_prime[i] == 0: # h_i 之前没有被更新过
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(lb_y[i]):
                        if e[i] * len(lb_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            lb_index = np.random.choice(len(lb_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            lb_X[i], lb_y[i] = lb_X[i][lb_index], lb_y[i][lb_index]
                            update[i] = True
            for i in range(3):
                print('### training estimator', i)
                if update[i]:
                    self.estimators[i].fit(np.append(X, lb_X[i], axis=0), np.append(y, lb_y[i], axis=0),
                                        eval_set=[(X_val, y_val)], verbose=self.verbose)
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])
            if update == [False] * 3:
                improve = False
            # if self.iter > 0:
            #     improve = False
        return self

    def predict_proba(self,X):
        y_proba = np.full((X.shape[0], 2), 0, np.float)
        for i in range(3):
            # y_proba+=self.estimators[i].predict_proba(X, num_iteration=self.estimators[i].best_iteration_)/3
            y_proba += self.estimators[i].predict_proba(X) / 3
        return y_proba

    def predict(self, X):
        # pred = np.asarray([self.estimators[i].predict(X, num_iteration=self.estimators[i].best_iteration_) for i in range(3)])
        pred = np.asarray([self.estimators[i].predict(X) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        y_pred=pred[0]
        return y_pred

    def measure_error(self, X, y, j, k):
        # j_pred = self.estimators[j].predict(X, num_iteration=self.estimators[j].best_iteration_)
        # k_pred = self.estimators[k].predict(X, num_iteration=self.estimators[k].best_iteration_)
        j_pred = self.estimators[j].predict(X)
        k_pred = self.estimators[k].predict(X)
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        return sum(wrong_index) / sum(j_pred == k_pred)


KF = StratifiedKFold(n_splits=kfold, random_state=2021, shuffle=True)
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

def get_model(kind):
    if kind == 'gbc':
        return GradientBoostingClassifier(max_depth=3)
    elif kind == 'hgbc':
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5
        )
    elif kind == 'xgbc':
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            # early_stopping_rounds=200,
            n_estimators=200,
            colsample_bytree=0.8,
            subsample=0.8,
            seed=2021,
        )
    elif kind == 'gbm':
        return LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            metrics='auc',
            n_jobs=30,
            learning_rate=0.05,
            num_leaves=2 ** 6,
            max_depth=8,
            tree_learner='serial',
            colsample_bytree=0.8,
            subsample_freq=1,
            subsample=0.8,
            num_boost_round=5000,
            max_bin=255,
            verbose=-1,
            seed=2021,
            bagging_seed=2021,
            feature_fraction_seed=2021,
            early_stopping_rounds=100,
            # n_estimators=200,
            # device='gpu',
            # gpu_platform_id=gpu_id,
            # gpu_device_id=gpu_id,
        )
    elif kind == 'cbc':
        return CatBoostClassifier(
            iterations=210,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=1,
            loss_function='Logloss',
            # early_stopping_rounds=100,
        )
    else:
        assert kind == 'fusion'
        gbc = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5
        )
        hgbc = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5
        )
        xgbc = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        gbm = LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            metrics='auc',
            n_jobs=30,
            learning_rate=0.05,
            num_leaves=2 ** 6,
            max_depth=8,
            tree_learner='serial',
            colsample_bytree=0.8,
            subsample_freq=1,
            subsample=0.8,
            num_boost_round=5000,
            max_bin=255,
            verbose=-1,
            seed=2021,
            bagging_seed=2021,
            feature_fraction_seed=2021,
            # early_stopping_rounds=100,
            n_estimators=200,
        )
        cbc = CatBoostClassifier(
            iterations=210,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=1,
            loss_function='Logloss',
        )
        estimators = [
            ('gbc', gbc),
            ('hgbc', hgbc),
            ('xgbc', xgbc),
            ('gbm', gbm),
            ('cbc', cbc)
        ]
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression()
        )
        return clf

def model_train(model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        # TT = Tri_Training(LGBMClassifier(**params), LGBMClassifier(**params), LGBMClassifier(**params))
        TT = Tri_Training(get_model(args.kind[0]), get_model(args.kind[1]), get_model(args.kind[2]))
        TT.fit(x_train.values, y_train.values, x_test.values,
               y_test.values, nolabel.values)

        y_pred = TT.predict_proba(x_test)[:, 1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test, y_pred)
        print("- KFold = %d, val_auc = %.8f" % (k, auc))
        test_fold_preds = TT.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.8f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold


clf_test_preds = model_train(kind, kfold)

submission['label'] = clf_test_preds
print(name)
submission.to_csv(name, index=False)
