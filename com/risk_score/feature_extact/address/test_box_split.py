import pandas as pd
import re
import time
import datetime
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from patsy.highlevel import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from numpy import log
from sklearn.metrics import roc_auc_score
import numpy as np
from  com.risk_score import scorecard_functions_V3 as sf


def box_split(train,test):

    cat_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'y']

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(train[var]))
        if valueCounts > 5:
            more_value_features.append(var)  # 取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)


    print(more_value_features)
    print(less_value_features)

    # （i）当取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量
    for col in less_value_features:
        #  bad rate ,某一个变量对应所有的标签必须涵盖两种标签，也就是单调
        binBadRate = sf.BinBadRate(train, col, 'y')[0]
        if min(binBadRate.values()) == 0:  # 由于某个取值没有坏样本而进行合并
            print('{} need to be combined due to 0 bad rate'.format(col))
            combine_bin = sf.MergeBad0(train, col, 'y')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            train[newVar] = train[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:  # 由于某个取值没有好样本而进行合并
            print('{} need to be combined due to 0 good rate'.format(col))
            combine_bin = sf.MergeBad0(train, col, 'y', direction='good')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            train[newVar] = train[col].map(combine_bin)
            var_bin_list.append(newVar)

    # 保存merge_bin_dict
    # with open('merge_bin_dict.pkl','wb+') as wf:
    #     pickle.dump(merge_bin_dict, wf)

    # less_value_features里剩下不需要合并的变量
    less_value_features = [i for i in less_value_features if i + '_Bin' not in var_bin_list]

    # 连续变量
    num_features = []
    # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
    br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
    for col in more_value_features:
        br_encoding = sf.BadRateEncoding(train, col, 'y')
        train[col + '_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']
        num_features.append(col + '_br_encoding')

    # 保存 br_encoding_dict
    # with open('br_encoding_dict.pkl','wb+') as wf:
    #     pickle.dump(br_encoding_dict, wf)

    # （iii）对连续型变量进行分箱，包括（ii）中的变量
    continous_merged_dict = {}
    for col in num_features:
        print("{} is in processing".format(col))
        if -1 not in set(train[col]):  # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
            max_interval = 5  # 分箱后的最多的箱数
            cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[], minBinPcnt=0)
            train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
            monotone = sf.BadRateMonotone(train, col + '_Bin', 'y')  # 检验分箱后的单调性是否满足
            while (not monotone):
                # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                max_interval -= 1
                cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[],
                                  minBinPcnt=0)
                train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
                if max_interval == 2:
                    # 当分箱数为2时，必然单调
                    break
                monotone = sf.BadRateMonotone(train, col + '_Bin', 'y')
            newVar = col + '_Bin'
            train[newVar] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
            var_bin_list.append(newVar)
        else:
            max_interval = 5
            # 如果有－1，则除去－1后，其他取值参与分箱
            cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[-1],
                              minBinPcnt=0)
            train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
            monotone = sf.BadRateMonotone(train, col + '_Bin', 'y', ['Bin -1'])
            while (not monotone):
                max_interval -= 1
                # 如果有－1，－1的bad rate不参与单调性检验
                cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                  minBinPcnt=0)
                train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
                if max_interval == 3:
                    # 当分箱数为3-1=2时，必然单调
                    break
                monotone = sf.BadRateMonotone(train, col + '_Bin', 'y', ['Bin -1'])
            newVar = col + '_Bin'
            train[newVar] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
            var_bin_list.append(newVar)
        continous_merged_dict[col] = cutOff

    # 保存 continous_merged_dict
    # with open('continous_merged_dict.pkl', 'wb+') as wf:
    #     pickle.dump(continous_merged_dict, wf)

    '''
    第四步：WOE编码、计算IV
    '''
    WOE_dict = {}
    IV_dict = {}
    # 分箱后的变量进行编码，包括：
    # 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中
    # 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 4，连续变量。分箱后新的变量存放在var_bin_list中
    all_var = var_bin_list + less_value_features
    for var in all_var:
        woe_iv = sf.CalcWOE(train, var, 'y')
        WOE_dict[var] = woe_iv['WOE']
        IV_dict[var] = woe_iv['IV']

    # 将变量IV值进行降序排列，方便后续挑选变量
    IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

    IV_values = [i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]
    plt.title('feature IV')
    plt.bar(range(len(IV_values)), IV_values)

    print('IV sort',IV_values)
    print('IV_name', IV_name)

    '''
    第五步：单变量分析和多变量分析，均基于WOE编码后的值。
    （1）选择IV高于0.01的变量
    （2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''

    # IV_dict.pop('loanamount_pdl_7_br_encoding_Bin')
    #
    # IV_dict.pop('loanamount_int_7_br_encoding_Bin')
    # IV_dict.pop('loan_avg_pdl_7_br_encoding_Bin')
    # #
    # IV_dict.pop('loan_avg_int_7_br_encoding_Bin')
    # IV_dict.pop('loanamount_pdl_14_br_encoding_Bin')
    # IV_dict.pop('apply_int_diff_11_br_encoding_Bin')
    # IV_dict.pop('approve_mert_pdl_diff_3_br_encoding_Bin')

    # 选取IV>0.01的变量
    high_IV = {k: v for k, v in IV_dict.items() if v >= 0.02}
    high_IV_sorted = sorted(high_IV.items(), key=lambda x: x[1], reverse=True)

    short_list = high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newVar = var + '_WOE'
        train[newVar] = train[var].map(WOE_dict[var])
        short_list_2.append(newVar)


    # 两两间的线性相关性检验
    # 1，将候选变量按照IV进行降序排列
    # 2，计算第i和第i+1的变量的线性相关系数
    # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
    deleted_index = []
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0] + "_WOE"
        for j in range(cnt_vars):
            if i == j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0] + "_WOE"
            roh = np.corrcoef(train[x1], train[y1])[0, 1]
            if abs(roh) > 0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)

    multi_analysis_vars_1 = [high_IV_sorted[i][0] + "_WOE" for i in range(cnt_vars) if i not in deleted_index]

    '''
    多变量分析：VIF
    '''
    X = np.matrix(train[multi_analysis_vars_1])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    max_VIF = max(VIF_list)
    print('多变量分析 maxVIF',max_VIF)
    # 最大的VIF是1.32267733123，因此这一步认为没有多重共线性
    multi_analysis = multi_analysis_vars_1

    '''
    第六步：逻辑回归模型。
    要求：
    1，变量显著
    2，符号为负
    '''
    ### (1)将多变量分析的后变量带入LR模型中
    multi_analysis = multi_analysis[:6]
    y = train['y']
    X = train[multi_analysis]
    X['intercept'] = [1] * X.shape[0]

    LR = sm.Logit(y, X).fit()
    summary = LR.summary()
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    # ### 有些变量不显著，需要逐步剔除
    # varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    # varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    # while (len(varLargeP) > 0 and len(multi_analysis) > 0):
    #     # 每次迭代中，剔除最不显著的变量，直到
    #     # (1) 剩余所有变量均显著
    #     # (2) 没有特征可选
    #     varMaxP = varLargeP[0][0]
    #     print(varMaxP)
    #     if varMaxP == 'intercept':
    #         print('the intercept is not significant!')
    #         break
    #     multi_analysis.remove(varMaxP)
    #     y = trainData['y']
    #     X = trainData[multi_analysis]
    #     X['intercept'] = [1] * X.shape[0]
    #
    #     LR = sm.Logit(y, X).fit()
    #     pvals = LR.pvalues
    #     pvals = pvals.to_dict()
    #     varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    #     varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    #
    # summary = LR.summary()
    # print(summary)

    print('入参变量',multi_analysis)
    print(X.shape)

    train['pred'] = LR.predict(X)
    ks = sf.KS(train, 'pred', 'y')
    # ks = sf.ks_calc_auc(train,train['pred'],train['y'])
    auc = roc_auc_score(train['y'], train['pred'])  # AUC = 0.73
    print('train 准确度Area Under Curve auc',auc,'区分度 KS',ks)

    # X_test = testData[multi_analysis]
    # print(X_test.shape)
    # X_test['intercept'] = [1] * X_test.shape[0]
    # testData['pred'] = LR.predict(X_test)
    # ks = sf.KS(testData, 'pred', 'y')
    # auc = roc_auc_score(testData['y'], testData['pred'])  # AUC = 0.73
    # print('Test 准确度', auc, 'Test 区分度 KS', ks)

    test_box_split(test, LR)


def test_box_split(train,LR):

    cat_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont != 'y']

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(train[var]))
        if valueCounts > 5:
            more_value_features.append(var)  # 取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)


    print(more_value_features)
    print(less_value_features)

    # （i）当取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量
    for col in less_value_features:
        #  bad rate ,某一个变量对应所有的标签必须涵盖两种标签，也就是单调
        binBadRate = sf.BinBadRate(train, col, 'y')[0]
        if min(binBadRate.values()) == 0:  # 由于某个取值没有坏样本而进行合并
            print('{} need to be combined due to 0 bad rate'.format(col))
            combine_bin = sf.MergeBad0(train, col, 'y')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            train[newVar] = train[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:  # 由于某个取值没有好样本而进行合并
            print('{} need to be combined due to 0 good rate'.format(col))
            combine_bin = sf.MergeBad0(train, col, 'y', direction='good')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            train[newVar] = train[col].map(combine_bin)
            var_bin_list.append(newVar)

    # 保存merge_bin_dict
    # with open('merge_bin_dict.pkl','wb+') as wf:
    #     pickle.dump(merge_bin_dict, wf)

    # less_value_features里剩下不需要合并的变量
    less_value_features = [i for i in less_value_features if i + '_Bin' not in var_bin_list]

    # 连续变量
    num_features = []
    # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
    br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
    for col in more_value_features:
        br_encoding = sf.BadRateEncoding(train, col, 'y')
        train[col + '_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']
        num_features.append(col + '_br_encoding')

    # 保存 br_encoding_dict
    # with open('br_encoding_dict.pkl','wb+') as wf:
    #     pickle.dump(br_encoding_dict, wf)

    # （iii）对连续型变量进行分箱，包括（ii）中的变量
    continous_merged_dict = {}
    for col in num_features:
        print("{} is in processing".format(col))
        if -1 not in set(train[col]):  # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
            max_interval = 5  # 分箱后的最多的箱数
            cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[], minBinPcnt=0)
            train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
            monotone = sf.BadRateMonotone(train, col + '_Bin', 'y')  # 检验分箱后的单调性是否满足
            while (not monotone):
                # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                max_interval -= 1
                cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[],
                                  minBinPcnt=0)
                train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
                if max_interval == 2:
                    # 当分箱数为2时，必然单调
                    break
                monotone = sf.BadRateMonotone(train, col + '_Bin', 'y')
            newVar = col + '_Bin'
            train[newVar] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
            var_bin_list.append(newVar)
        else:
            max_interval = 5
            # 如果有－1，则除去－1后，其他取值参与分箱
            cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[-1],
                              minBinPcnt=0)
            train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
            monotone = sf.BadRateMonotone(train, col + '_Bin', 'y', ['Bin -1'])
            while (not monotone):
                max_interval -= 1
                # 如果有－1，－1的bad rate不参与单调性检验
                cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                  minBinPcnt=0)
                train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
                if max_interval == 3:
                    # 当分箱数为3-1=2时，必然单调
                    break
                monotone = sf.BadRateMonotone(train, col + '_Bin', 'y', ['Bin -1'])
            newVar = col + '_Bin'
            train[newVar] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
            var_bin_list.append(newVar)
        continous_merged_dict[col] = cutOff

    # 保存 continous_merged_dict
    # with open('continous_merged_dict.pkl', 'wb+') as wf:
    #     pickle.dump(continous_merged_dict, wf)

    '''
    第四步：WOE编码、计算IV
    '''
    WOE_dict = {}
    IV_dict = {}
    # 分箱后的变量进行编码，包括：
    # 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中
    # 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 4，连续变量。分箱后新的变量存放在var_bin_list中
    all_var = var_bin_list + less_value_features
    for var in all_var:
        woe_iv = sf.CalcWOE(train, var, 'y')
        WOE_dict[var] = woe_iv['WOE']
        IV_dict[var] = woe_iv['IV']

    # 将变量IV值进行降序排列，方便后续挑选变量
    IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

    IV_values = [i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]
    plt.title('feature IV')
    plt.bar(range(len(IV_values)), IV_values)

    print('IV sort',IV_values)
    print('IV_name', IV_name)

    '''
    第五步：单变量分析和多变量分析，均基于WOE编码后的值。
    （1）选择IV高于0.01的变量
    （2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''

    # IV_dict.pop('loanamount_pdl_7_br_encoding_Bin')
    #
    # IV_dict.pop('loanamount_int_7_br_encoding_Bin')
    # IV_dict.pop('loan_avg_pdl_7_br_encoding_Bin')
    # #
    # IV_dict.pop('loan_avg_int_7_br_encoding_Bin')
    # IV_dict.pop('loanamount_pdl_14_br_encoding_Bin')
    # IV_dict.pop('apply_int_diff_11_br_encoding_Bin')
    # IV_dict.pop('approve_mert_pdl_diff_3_br_encoding_Bin')

    # 选取IV>0.01的变量
    high_IV = {k: v for k, v in IV_dict.items() if v >= 0.02}
    high_IV_sorted = sorted(high_IV.items(), key=lambda x: x[1], reverse=True)

    short_list = high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newVar = var + '_WOE'
        train[newVar] = train[var].map(WOE_dict[var])
        short_list_2.append(newVar)


    # 两两间的线性相关性检验
    # 1，将候选变量按照IV进行降序排列
    # 2，计算第i和第i+1的变量的线性相关系数
    # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
    deleted_index = []
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0] + "_WOE"
        for j in range(cnt_vars):
            if i == j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0] + "_WOE"
            roh = np.corrcoef(train[x1], train[y1])[0, 1]
            if abs(roh) > 0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)

    multi_analysis_vars_1 = [high_IV_sorted[i][0] + "_WOE" for i in range(cnt_vars) if i not in deleted_index]

    '''
    多变量分析：VIF
    '''
    X = np.matrix(train[multi_analysis_vars_1])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    max_VIF = max(VIF_list)
    print('多变量分析 maxVIF',max_VIF)
    # 最大的VIF是1.32267733123，因此这一步认为没有多重共线性
    multi_analysis = multi_analysis_vars_1

    '''
    第六步：逻辑回归模型。
    要求：
    1，变量显著
    2，符号为负
    '''
    ### (1)将多变量分析的后变量带入LR模型中
    multi_analysis = multi_analysis[:6]

    y = train['y']
    X = train[multi_analysis]
    X['intercept'] = [1] * X.shape[0]


    print('入参变量',multi_analysis)
    print(X.shape)

    train['pred'] = LR.predict(X)
    ks = sf.KS(train, 'pred', 'y')
    # ks = sf.ks_calc_auc(train,train['pred'],train['y'])
    auc = roc_auc_score(train['y'], train['pred'])  # AUC = 0.73
    print('test 准确度Area Under Curve auc',auc,'区分度 KS',ks)
