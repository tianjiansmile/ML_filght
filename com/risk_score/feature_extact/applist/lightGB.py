import pandas as pd
import jieba
from gensim import corpora, models
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import ks_2samp
import lightgbm as lgb
from  com.risk_score import scorecard_functions_V3 as sf
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import model_selection
import numpy as np


def lgb_model(X_train, X_valid, y_train, y_valid):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.001,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': -1,
        'num_threads': 20}
    model = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=50000,
                    valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=4000,
                    verbose_eval=400)
    #model.save_model('/home/wangyuanjiang/thirdpartydataTest/suanhua/model/'+model_name+'.model')
    return(model)

def getAUC(model, model_column, dfTrain_X, y_train, dfTest_X, y_test, dfTime_X):

    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_auc = roc_auc_score(y_train.astype("int"), data_predict_1train)
    data_predict_1test = model.predict(dfTest_X.astype("float"))
    test_auc = roc_auc_score(y_test.astype("int"), data_predict_1test)
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_auc = roc_auc_score(dfTime_X.overdue_days.astype("int"), data_predict_1time)
    print("训练上的auc：%f \n验证集上的auc：%f \n测试集上的auc：%f "%(train_auc, test_auc, time_auc))

def compute_ks(model, model_column, dfTrain_X, y_train, dfTest_X, y_test, dfTime_X):
    '''
    target: numpy array of shape (1,)
    proba: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation

    '''
    get_ks = lambda proba, target: ks_2samp(proba[target == 1], proba[target != 1]).statistic
    target = dfTime_X.overdue_days.astype("int")
    data_predict_1train = model.predict(dfTrain_X.astype("float"))
    train_ks = get_ks(data_predict_1train, y_train.astype("int"))
    data_predict_1test = model.predict(dfTest_X.astype("float"))
    test_ks = get_ks(data_predict_1test, y_test.astype("int"))
    data_predict_1time = model.predict(np.array(dfTime_X[model_column].astype("float")))
    time_ks = get_ks(data_predict_1time, dfTime_X.overdue_days.astype("int"))

    print("训练上的KS：%f \n测试上的KS：%f \n跨时间上的KS：%f " % (train_ks, test_ks, time_ks))

def feature_importance(model, column_name):
    """get_top_10_feature_importance
       param model:model
    """
    feat_imp = pd.DataFrame(
        dict(columns=column_name, feature_importances=model.feature_importance() / model.feature_importance().sum()))
    feat_imp2 = feat_imp.sort_values(by=["feature_importances"], axis=0, ascending=False).head(100)
    return (feat_imp2)

def label_map(x):
    if int(x) >= 10:
        return 1
    else:
        return 0

def train(trainData, testData, col):

    X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(np.array(trainData[col]),
                                                                          trainData['overdue_days'].values,
                                                                          test_size=0.20,
                                                                          random_state=123)

    model = lgb_model(X_train, X_valid, Y_train, Y_valid)

    getAUC(model,
           col,
           X_train,
           Y_train,
           X_valid,
           Y_valid,
           testData)

    compute_ks(model,
               col,
               X_train,
               Y_train,
               X_valid,
               Y_valid,
               testData)

    fea_impt = feature_importance(model, col)[:20]
    print(fea_impt)


def miaola_app():
    allData = pd.read_excel('topics350.xlsx', sheetname='sheet1')

    # 暂时删除有空数据的行
    allData.dropna(axis=0, how='any', inplace=True)

    # 确保二分类
    allData['overdue_days'] = allData['overdue_days'].map(label_map)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'overdue_days']

    trainData, testData = train_test_split(allData, test_size=0.23)

    train_rate = trainData.overdue_days.sum() / trainData.shape[0]
    test_rate = testData.overdue_days.sum() / testData.shape[0]

    print('train_rate: ', train_rate, ' test_rate: ', test_rate)

    # 训练
    train(trainData, testData, cat_features)

def miaola_addr():
    allData = pd.read_excel('../address/秒啦首贷_train_pd10.xlsx', sheetname='Sheet1')

    # 暂时删除有空数据的行
    allData.dropna(axis=0, how='any', inplace=True)

    # 将不参与训练的特征数据删除
    allData.drop(['live_addr'], axis=1, inplace=True)

    # 确保二分类
    allData['overdue_days'] = allData['overdue_days'].map(label_map)

    cat_features = [cont for cont in list(allData.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'overdue_days']

    trainData, testData = train_test_split(allData, test_size=0.33,random_state=43)

    train_rate = trainData.target.sum() / trainData.shape[0]
    test_rate = testData.target.sum() / testData.shape[0]

    print('train_rate: ',train_rate,' test_rate: ',test_rate)

    # 训练
    train(trainData, testData, cat_features)
if __name__ == '__main__':
    miaola_app()

    # miaola_addr()