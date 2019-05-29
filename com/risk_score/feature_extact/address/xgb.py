# First XGBoost model for Pima Indians dataset
# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from  com.risk_score import scorecard_functions_V3 as sf
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

def label_map(x):
    if int(x) >= 10:
        return 1
    else:
        return 0

def train(trainData,testData,col):

    X_train = trainData[col]
    Y_train = trainData['overdue_days']

    # fit model no training data
    # model = XGBClassifier()

    # model = XGBClassifier()

    model = XGBClassifier(learning_rate=0.1, min_samples_split=30,
                          min_samples_leaf=5, max_depth=6, max_features='sqrt',
                          subsample=0.8, random_state=10)

    # 训练集，对没新加入的树进行测试
    eval_set = [(X_train, Y_train)]

    # early_stopping_rounds 连续10次loss不下降停止模型，
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_train)
    trainData['pred'] =y_pred
    y_predprob = model.predict_proba(X_train)[:, 1].T
    trainData['predprob'] = y_predprob
    
    print(trainData['overdue_days'].dtype,trainData['pred'].dtype)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_train, predictions)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (Train): %f" % metrics.roc_auc_score(trainData['overdue_days'], y_predprob))

    ks = sf.KS(trainData, 'predprob', 'overdue_days')
    # ks = ks_calc_auc(trainData, 'pred', 'overdue_sum_label')
    print('Train KS:', ks)



    Y_test = testData['overdue_days']
    X_test = testData[col]
    y_test_pred = model.predict(X_test)
    y_test_predprob = model.predict_proba(X_test)[:, 1].T
    testData['predprob'] = y_test_predprob

    test_predictions = [round(value) for value in y_test_pred]
    accuracy = accuracy_score(Y_test, test_predictions)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
    testData['pred'] = y_test_pred

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (test): %f" % metrics.roc_auc_score(np.array(Y_test.T), y_test_predprob))

    ks = sf.KS(testData, 'predprob', 'overdue_days')
    print('Test KS:', ks)

    plot_importance(model)
    pyplot.show()

    # time_test(model)

def time_test(model):
    file = '秒啦首贷_timetest_pd10.xlsx'
    train = pd.read_excel(file, sheetname='Sheet1')

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将逾期次数转化为0，1标签
    train['y'] = train['overdue_days'].map(label_map)

    print(len(train['y'].unique()))

    # 将不参与训练的特征数据删除
    train.drop(['live_addr', 'overdue_days'], axis=1, inplace=True)

    col = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'y']

    Y_test = train['y']
    X_test = train[col]
    y_test_pred = model.predict(X_test)
    y_test_predprob = model.predict_proba(X_test)[:, 1].T
    train['predprob'] = y_test_predprob

    test_predictions = [round(value) for value in y_test_pred]
    accuracy = accuracy_score(Y_test, test_predictions)
    print("time Test Accuracy: %.2f%%" % (accuracy * 100.0))
    train['pred'] = y_test_pred

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (time test): %f" % metrics.roc_auc_score(np.array(Y_test.T), y_test_predprob))

    ks = sf.KS(train, 'predprob', 'y')
    print('time Test KS:', ks)


def miaola_test():
    allData = pd.read_excel('秒啦首贷_train_pd10.xlsx', sheetname='Sheet1')

    # 暂时删除有空数据的行
    allData.dropna(axis=0, how='any', inplace=True)

    # 将不参与训练的特征数据删除
    allData.drop(['live_addr'], axis=1, inplace=True)


    # 确保二分类
    allData['overdue_days'] = allData['overdue_days'].map(label_map)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'overdue_days' ]


    trainData, testData = train_test_split(allData, test_size=0.33)

    # 训练
    train(trainData, testData,cat_features)

def miaola_app_test():
    allData = pd.read_excel('../applist/topics.xlsx', sheetname='sheet1')

    # 暂时删除有空数据的行
    allData.dropna(axis=0, how='any', inplace=True)


    # 确保二分类
    allData['overdue_days'] = allData['overdue_days'].map(label_map)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'overdue_days']

    trainData, testData = train_test_split(allData, test_size=0.33)

    # 训练
    train(trainData, testData, cat_features)


if __name__ == '__main__':
    # miaola_test()

    miaola_app_test()