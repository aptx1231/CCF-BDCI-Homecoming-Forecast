import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

kind = 'fusion'
print(kind)

train_data = pd.read_csv('data/dataTrain.csv')  # train_df (59872, 48)
test_data = pd.read_csv('data/dataB.csv')
submission = pd.read_csv('data/submit_example_B.csv')
data_nolabel = pd.read_csv('data/dataNoLabel.csv')
print(f'origin train_data.shape = {train_data.shape}')
print(f'origin test_data.shape  = {test_data.shape}')


# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)

# 离散化
all_f = [f'f{idx}' for idx in range(1, 47) if idx != 3]
for df in [train_data, test_data]:
    for col in all_f:
        df[f'{col}_log'] = df[col].apply(lambda x: int(np.log(x)) if x > 0 else 0)

# 特征交叉
log_f = [f'f{idx}_log' for idx in range(1, 47) if idx != 3]
for df in [train_data, test_data]:
    for i in range(len(log_f)):
        for j in range(i + 1, len(log_f)):
            df[f'{log_f[i]}_{log_f[j]}'] = df[log_f[i]]*10000 + df[log_f[j]]

# 离散化f3
cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])

print(f'processed train_data.shape = {train_data.shape}')
print(f'processed test_data.shape  = {test_data.shape}')

num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

print(feature_columns)
train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]
print(train.shape, label.shape, test.shape)

train = train[:50000]
label = label[:50000]
print(train.shape, label.shape, test.shape)


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
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
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
            n_estimators=200,
        )
    elif kind == 'cbc':
        return CatBoostClassifier(
            iterations=210,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=1,
            loss_function='Logloss',
        )
    else:
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
            num_leaves=2 ** 6,
            max_depth=8,
            colsample_bytree=0.8,
            subsample_freq=1,
            max_bin=255,
            learning_rate=0.05,
            n_estimators=100,
            metrics='auc'
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


def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        # model.fit(x_train, y_train, eval_set=(x_test, y_test))
        model.fit(x_train, y_train)

        y_pred = model.predict_proba(x_test)[:, 1]
        # y_pred = model.predict(x_test)
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test, y_pred)
        print("- KFold = %d, val_auc = %.8f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.8f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold


model = get_model(kind)

# 特征筛选
X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)

# model.fit(X_train, y_train, eval_set=(X_test, y_test))
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('origin auc = %.8f' % auc)

ff = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        ff.append(col)  # 记录使得AUC降低的特征
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc))

model.fit(X_train[ff], y_train)
# model.fit(X_train[ff], y_train, eval_set=(X_test[ff], y_test))
y_pred = model.predict_proba(X_test[ff])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('select auc = %.8f' % auc)
print('select features', ff)

train = train[ff]
test = test[ff]

clf_test_preds = model_train(model, kind, 10)

submission['label'] = clf_test_preds
name = 'res/submission-base2-B{}-log.csv'.format(kind)
print(name)
submission.to_csv(name, index=False)
