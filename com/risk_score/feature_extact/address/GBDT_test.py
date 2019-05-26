import pandas as pd
import pickle
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import  metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from  com.risk_score import scorecard_functions_V3 as sf
import matplotlib.pylab as plt
import time
import datetime
from dateutil.relativedelta import relativedelta
from numpy import log
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model.logistic import LogisticRegression

# 将用户审核通过次数转换为是否审核通过
def map_label(x):
    if x >= 10:
        return 1
    else:
        return 0

def tow_label(x):
    if x == 0:
        return 0
    else:
        return 1

def train_test(train):

    trainData, testData = train_test_split(train, test_size=0.3)

    cat_features = [cont for cont in list(trainData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'y']

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(trainData[var]))
        if valueCounts > 5:
            more_value_features.append(var)
        else:
            less_value_features.append(var)

    print('num', more_value_features)
    print('cat', less_value_features)

    v = DictVectorizer(sparse=False)
    X2 = np.matrix(trainData[more_value_features])
    if len(less_value_features) > 0:
        X1 = v.fit_transform(trainData[less_value_features].to_dict('records'))
        # 将独热编码和数值型变量放在一起进行模型训练
        X = np.hstack([X1, X2])
    else:
        X = X2

    y = trainData['y']
    # 未经调参进行GBDT模型训练
    gbm0 = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=30,
                                                              min_samples_leaf=5, max_depth=5, max_features='sqrt',
                                                              subsample=0.8, random_state=10)

    # gbm0 = GradientBoostingClassifier(random_state=10)

    gbm0.fit(X, y)

    y_pred = gbm0.predict(X)
    y_predprob = gbm0.predict_proba(X)[:, 1].T
    trainData['pred'] = y_predprob
    print("Accuracy : %.4g" % metrics.accuracy_score(y, y_pred))
    ks = sf.KS(trainData, 'pred', 'y')
    print("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(y.T), y_predprob))
    print('KS :',ks)

    # 在测试集上测试效果
    cat_features = [cont for cont in list(testData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'y']

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(testData[var]))
        if valueCounts > 5:
            more_value_features.append(var)  # 取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)

    print('num', more_value_features)
    print('cat', less_value_features)

    v = DictVectorizer(sparse=False)

    X2 = np.matrix(testData[more_value_features])
    if len(less_value_features) > 0:
        X1 = v.fit_transform(testData[less_value_features].to_dict('records'))
        # 将独热编码和数值型变量放在一起进行模型训练
        X_test = np.hstack([X1, X2])
    else:
        X_test = X2

    y_test = np.matrix(testData['y']).T

    # 在测试集上测试GBDT性能
    y_pred_test = gbm0.predict(X_test)
    y_predprob_test = gbm0.predict_proba(X_test)[:, 1].T
    testData['pred'] = y_predprob_test
    testData['predprob'] = list(y_predprob_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:, 0], y_predprob_test))
    ks = sf.KS(testData, 'pred', 'y')
    print('Test KS :',ks)

    # 在跨时间数据集测试
    time_test(gbm0)


def time_test(gbm):
    file = '秒啦首贷_timetest_pd10.xlsx'
    train = pd.read_excel(file, sheetname='Sheet1')

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将逾期次数转化为0，1标签
    train['y'] = train['overdue_days'].map(map_label)

    print(len(train['y'].unique()))

    # 将不参与训练的特征数据删除
    train.drop(['live_addr', 'overdue_days'], axis=1, inplace=True)

    # 变量类型超过5
    more_value_features = []
    less_value_features = []

    cat_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'y']


    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(train[var]))
        if valueCounts > 5:
            more_value_features.append(var)
        else:
            less_value_features.append(var)

    v = DictVectorizer(sparse=False)
    X2 = np.matrix(train[more_value_features])
    if len(less_value_features) > 0:
        X1 = v.fit_transform(train[less_value_features].to_dict('records'))
        # 将独热编码和数值型变量放在一起进行模型训练
        X = np.hstack([X1, X2])
    else:
        X = X2

    y = train['y']

    y_pred = gbm.predict(X)
    y_predprob = gbm.predict_proba(X)[:, 1].T
    train['pred'] = y_predprob
    print("跨时间 Accuracy : %.4g" % metrics.accuracy_score(y, y_pred))
    print("AUC Score (夸时间): %f" % metrics.roc_auc_score(np.array(y.T), y_predprob))
    ks = sf.KS(train, 'pred', 'y')
    print('Time Test KS :', ks)


def params_adjust(train):
    cat_features = [cont for cont in list(train.select_dtypes(
     include=['float64', 'int64']).columns) if cont != 'y']

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
     valueCounts = len(set(train[var]))
     if valueCounts > 5:
         more_value_features.append(var)
     else:
         less_value_features.append(var)

    print('num', more_value_features)
    print('cat', less_value_features)

    v = DictVectorizer(sparse=False)
    X2 = np.matrix(train[more_value_features])
    if len(less_value_features) > 0:
     X1 = v.fit_transform(train[less_value_features].to_dict('records'))
     # 将独热编码和数值型变量放在一起进行模型训练
     X = np.hstack([X1, X2])
    else:
     X = X2

    y = train['y']

    # 1, 选择较小的步长(learning rate)后，对迭代次数(n_estimators)进行调参
    param_test1 = {'n_estimators': range(20, 81, 10)}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=30,
                                                              min_samples_leaf=5, max_depth=8, max_features='sqrt',
                                                              subsample=0.8, random_state=10),
                         param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    gsearch1.fit(X, y)
    gsearch1.best_params_, gsearch1.best_score_
    best_n_estimator = gsearch1.best_params_['n_estimators']

    # 2, 对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
    param_test2 = {'max_depth': range(3, 16), 'min_samples_split': range(2, 10)}
    gsearch2 = GridSearchCV(
     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator, min_samples_leaf=20,
                                          max_features='sqrt', subsample=0.8, random_state=10),
     param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(X, y)
    gsearch2.best_params_, gsearch2.best_score_
    best_max_depth = gsearch2.best_params_['max_depth']

    # 3, 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
    param_test3 = {'min_samples_split': range(10, 101, 10), 'min_samples_leaf': range(5, 51, 5)}
    gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,
                                                              max_depth=best_max_depth,
                                                              max_features='sqrt', subsample=0.8, random_state=10),
                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    gsearch3.fit(X, y)
    gsearch3.best_params_, gsearch3.best_score_
    best_min_samples_split, best_min_samples_leaf = gsearch3.best_params_['min_samples_split'], gsearch3.best_params_[
     'min_samples_leaf']

    # 4, 对最大特征数max_features进行网格搜索
    param_test4 = {'max_features': range(5, int(np.sqrt(X.shape[0])), 5)}
    gsearch4 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,
                                                              max_depth=best_max_depth,
                                                              min_samples_leaf=best_min_samples_leaf,
                                                              min_samples_split=best_min_samples_split,
                                                              subsample=0.8, random_state=10),
                         param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
    gsearch4.fit(X, y)
    gsearch4.best_params_, gsearch4.best_score_
    best_max_features = gsearch4.best_params_['max_features']

    # 5, 对采样比例进行网格搜索
    param_test5 = {'subsample': [0.6 + i * 0.05 for i in range(8)]}
    gsearch5 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,
                                                              max_depth=best_max_depth,
                                                              min_samples_leaf=best_min_samples_leaf,
                                                              max_features=best_max_features, random_state=10),
                         param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)

    print('best_n_estimator',best_n_estimator)
    print('best_max_depth',best_max_depth)
    print('best_min_samples_split',best_min_samples_split, 'best_min_samples_leaf',best_min_samples_leaf)
    print('best_max_features',best_max_features)
    print('gsearch5',gsearch5)

    gsearch5.fit(X, y)
    gsearch5.best_params_, gsearch5.best_score_
    best_subsample = gsearch5.best_params_['subsample']

    gbm_best = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator, max_depth=best_max_depth,
                                       min_samples_leaf=best_min_samples_leaf, max_features=best_max_features,
                                       subsample=best_subsample, random_state=10)
    gbm_best.fit(X, y)

    # 在测试集上测试并计算性能
    y_pred = gbm_best.predict(X)
    y_predprob = gbm_best.predict_proba(X)[:, 1].T
    print("Accuracy : %.4g" % metrics.accuracy_score(y, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(y.T), y_predprob))


if __name__ == '__main__':
    file = '秒啦首贷_train_pd10.xlsx'
    # file = 'approve_addr_feature_train.xlsx'
    train = pd.read_excel(file, sheetname='Sheet1')

    # data_check(train)

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将逾期次数转化为0，1标签
    train['y'] = train['overdue_days'].map(map_label)
    # train['y'] = train['overdue_days'].map(tow_label)

    print(len(train['y'].unique()))

    # 将不参与训练的特征数据删除
    train.drop(['live_addr', 'overdue_days'], axis=1, inplace=True)

    # 训练并测试
    train_test(train)

    # 参数调节
    # params_adjust(train)
