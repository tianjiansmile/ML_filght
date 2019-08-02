#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'tianjian'

import time
import datetime
import numpy as np
import difflib


# 本脚本侧重提供对于pandas 数据的各种分组切片
# 逾期天数的计算等

loan_order = ['apply', 'reject', 'approve', 'loanamount',
              'approve_rate', 'reject_rate', 'loan_rate','loan_avg', 'loan_times',
              'overdue', 'pd7', 'pd14', 'M1','maxOverdue','avg_overdueday','overdue_rate']

loan_repay = ['pd0', 'pd7', 'pd14', 'M1','max_overdueday','avg_overdueday']

loan_type = ['pdl', 'int', 'sum']

lh_day_map = {'EM': 'earlyMorning',
            'M': 'morning',
            'NO': 'noon',
            'AF': 'afternoon',
            'E': 'evening',
            'NI': 'night'}

feature_default_value = -1
feature_no_order = 'NaN'

lh_week_map = {'WD': 'weekday', 'WE': 'weekend'}

loan_monthly = [1,3,7,14,30,60,90,180,365,999]

loan_merchant_monthly = [1,2,3,4,5,6,7,8,9,10,11,12,13]

lh_month_map = {
            1: 'last1',
            3: 'last3',
            7: 'last7',
            14: 'last14',
            30: 'last30',
            60: 'last60',
            90: 'last90',
            180: 'last180',
            365: 'last356',
            999: 'lastall'
}


def check_overdueday_pre(repay_state,repay_date,actual_date,merchant_id):

    overdureday = None
    second = 86400
    # 你我贷贷后超过28小时算作逾期
    second_pro = 100800

    # 获取当前日期
    now_date = datetime.datetime.now().date()

    repays_date = repay_date.date()

    if repay_state == '2': #还款标志
        if actual_date:

            actuals_date = actual_date.date()
            overdureday = (actuals_date - repays_date).days
            if overdureday == 1:
                overduretime = actual_date - repay_date
                if merchant_id == '11':
                    if overduretime.seconds < second: # 小于一天
                        overdureday = 0
                else:
                    if overduretime.seconds < second_pro:  # 小于一天
                        overdureday = 0


        else:
            overdureday = (now_date - repays_date).days
            if overdureday == 1:
                overduretime = datetime.datetime.now() - repay_date
                if merchant_id == '11':
                    if overduretime.seconds < second:  # 小于一天
                        overdureday = 0
                else:
                    if overduretime.seconds < second_pro:  # 小于一天
                        overdureday = 0
    else:
        overdureday = (now_date - repays_date).days
        if overdureday == 1:
            overduretime = datetime.datetime.now() - repay_date
            if merchant_id == '11':
                if overduretime.seconds < second:  # 小于一天
                    overdureday = 0
            else:
                if overduretime.seconds < second_pro:  # 小于一天
                    overdureday = 0

    if overdureday < 0:
        overdureday = 0
    return overdureday

def check_overdueday(repay_state,repay_date,actual_date,merchant_id):

    overdureday = None

    # 获取当前日期
    now_time = datetime.datetime.now()
    now_date = datetime.datetime.now().date()

    repays_date = repay_date.date()

    if repay_state == '2': #还款标志
        if actual_date:

            actuals_date = actual_date.date()
            overdureday = (actuals_date - repays_date).days
            if overdureday == 1:
                if merchant_id == '11':
                    hour = actual_date.hour
                    niwodai = 4
                    if hour < niwodai:
                        overdureday = 0
                    # print('你我贷：', hour)


        else:
            overdureday = (now_date - repays_date).days
            if overdureday == 1:
                if merchant_id == '11':
                    hour = now_time.hour
                    niwodai = 4
                    if hour < niwodai:
                        overdureday = 0
                    # print('你我贷：', hour)
    else:
        overdureday = (now_date - repays_date).days
        if overdureday == 1:
            if merchant_id == '11':
                hour = now_time.hour
                niwodai = 4
                if hour < niwodai:
                    overdureday = 0
                # print('你我贷：',hour)

    # print('overdureday',overdureday)
    if overdureday < 0:
        overdureday = 0
    return overdureday

def check_overdueday_timeback(repay_state,repay_date,actual_date,merchant_id,timeback):

    overdureday = None

    # 获取当前日期
    now_date = timeback.date()
    if actual_date:
        actuals_date = actual_date.date()
    if repay_date:
        repays_date = repay_date.date()

    if repay_state == '2': #还款标志
        if (now_date - repays_date).days > 0:
            if actual_date:
                if (now_date - actuals_date).days > 0:
                    overdureday = (actuals_date - repays_date).days
                    if overdureday == 1:
                        if merchant_id == '11':
                            hour = actual_date.hour
                            niwodai = 4
                            if hour < niwodai:
                                overdureday = 0
                            # print('你我贷：', hour)
                else:
                    overdureday = (now_date - repays_date).days
                    if overdureday == 1:
                        if merchant_id == '11':
                            hour = actual_date.hour
                            niwodai = 4
                            if hour < niwodai:
                                overdureday = 0
                            # print('你我贷：', hour)


            else:
                overdureday = (now_date - repays_date).days
                if overdureday == 1:
                    if merchant_id == '11':
                        hour = timeback.hour
                        niwodai = 4
                        if hour < niwodai:
                            overdureday = 0
                        # print('你我贷：', hour)
        else:
            overdureday = 0
    else:
        overdureday = (now_date - repays_date).days
        if overdureday == 1:
            if merchant_id == '11':
                hour = timeback.hour
                niwodai = 4
                if hour < niwodai:
                    overdureday = 0
                # print('你我贷：',hour)

    # print('overdureday',overdureday)
    if overdureday < 0:
        overdureday = 0
    return overdureday

def diff_time(data1,data2):

    return data1 - data2

def time_second_diff(curr_time,pre_time):
    oneday_second = 86400

    if pre_time > curr_time:
        time_diff = pre_time - curr_time
    else:
        time_diff = curr_time - pre_time

    # time_diff = pre_time - curr_time

    return time_diff.seconds,time_diff.days

def feature_init(column, martix,have_order=1):
    user_feature = dict()
    loan_v = loan_order
    loan_p = loan_type

    v1, v2 = np.shape(martix)
    for i in range(v1):
        for j in range(v2):
            key = loan_v[i] + '_' + loan_p[j] + '_' + str(column)  # 组装 key
            if i == 3:
                temp = martix[i][j]  # 赋值
            else:
                temp = round(float(martix[i][j]),2)

            if have_order == 1:
                user_feature[key] = temp
            else:
                # user_feature[key] = feature_default_value
                user_feature[key] = feature_no_order

    return user_feature

def apply_time_window(dt, colname, cutoff_time):
    """
    add more windows to dt.col
    1. TW_1: morning, afternoon, evening, night
    2. TW_2: weekday, weekend
    4. TW_3: L7D, L14D, L30D, L45D, L60D, L90D, L180D # NOTE: last N days from cutoff_time
    3. TW_4: holiday, non_holiday # todo

    :param dt: data frame
    :param colname: column name
    :param cutoff_time: time stamp
    """
    ls = list(dt[colname])
    ls_tw_1 = list(map(fun_tw_1, ls))
    ls_tw_2 = list(map(fun_tw_2, ls))
    ls_tw_3 = list(map(fun_tw_3, ls, [cutoff_time] * len(ls)))
    ls_tw_4 = list(map(fun_tw_4, ls, [cutoff_time] * len(ls)))
    dt = dt.assign(TW_1=ls_tw_1,
                   TW_2=ls_tw_2,
                   TW_3=ls_tw_3,
                   TW_4=ls_tw_4)
    return dt

def apply_repay_time_window(dt, colname, cutoff_time):
    """
    :param cutoff_time: time stamp
    """
    ls = list(dt[colname])
    ls_tw_1 = list(map(fun_tw_1, ls))
    ls_tw_2 = list(map(fun_tw_2, ls))
    ls_tw_3 = list(map(fun_tw_3, ls, [cutoff_time] * len(ls)))
    ls_tw_4 = list(map(fun_tw_4, ls, [cutoff_time] * len(ls)))
    dt = dt.assign(TW_1=ls_tw_1,
                   TW_2=ls_tw_2,
                   TW_3=ls_tw_3,
                   TW_4=ls_tw_4)
    return dt

def apply_type_window(dt, colname):
    """
    :param cutoff_time: time stamp
    """
    ls = list(dt[colname])
    ls_tw_3 = list(map(fun_type, ls))
    dt = dt.assign(p_type=ls_tw_3)
    return dt

def fun_type(t):
    if t == '2':
        return 'int'
    else:
        return 'pdl'

def fun_tw_3(t, cutoff_time):
    """
    # L7D, L14D, L30D, L45D, L60D, L90D, L180D # NOTE: last N days from cutoff_time
    :param t: a timestamp
    :param cutoff_time: a timestamp
    """
    seconds_val = (cutoff_time - t).total_seconds()
    val = seconds_val / 86400
    if seconds_val > 0:
        if val <= 1:
            return 1
        elif val <= 3:
            return 3
        elif val <= 7:
            return 7
        elif val <= 14:
            return 14
        elif val <= 30:
            return 30
        # elif val <= 45:
        #     return 45
        elif val <= 60:
            return 60
        elif val <= 90:
            return 90
        elif val <= 180:
            return 180
        elif val <=365:
            return 365
        elif val > 365:
            return 999


def fun_tw_4(t, cutoff_time):
    """
    # L7D, L14D, L30D, L45D, L60D, L90D, L180D # NOTE: last N days from cutoff_time
    :param t: a timestamp
    :param cutoff_time: a timestamp
    """
    seconds_val = (cutoff_time - t).total_seconds()
    val = seconds_val / 86400
    if seconds_val > 0:
        if val >= 360:
            return 13
        elif val >= 330:
            return 12
        elif val >=300:
            return 11
        elif val >= 270:
            return 10
        elif val >=240:
            return 9
        elif val >= 210:
            return 8
        elif val >= 180:
            return 7
        elif val >= 150:
            return 6
        elif val >= 120:
            return 5
        elif val >= 90:
            return 4
        elif val >= 60:
            return 3
        elif val >= 30:
            return 2
        elif val >= 0:
            return 1

def fun_tw_2(t):
    """
    # WD: weekday: 0, 1, 2, 3, 4
    # WE: weekend: 5, 6
    :param t: a timestamp
    """
    if t.weekday() in [5, 6]:
        return 'WE'
    else:
        return 'WD'

def fun_tw_1(t):
    """
    # EM: early_morning: 5, 6, 7
    # M: morning: 8, 9, 10, 11
    # NO: noon: 12, 13
    # A: afternoon: 14, 15, 16, 17, 18
    # E: evening: 19, 20, 21, 22, 23
    # NI: night: 23, 0, 1, 2, 3, 4
    :param t: a timestamp
    """
    if t.hour in [5, 6, 7]:
        return 'EM'
    elif t.hour in [8, 9, 10, 11]:
        return 'M'
    elif t.hour in [12, 13]:
        return 'NO'
    elif t.hour in [14, 15, 16, 17, 18]:
        return 'AF'
    elif t.hour in [19, 20, 21, 22, 23]:
        return 'E'
    else:
        return 'NI'

def fun_interval_split(val,apply_interval_dict):
    """
    # L7D, L14D, L30D, L45D, L60D, L90D, L180D # NOTE: last N days from cutoff_time
    :param t: a timestamp
    :param cutoff_time: a timestamp
    """
    if val <= 1:
        apply_interval_dict['apply_interval_0_1']+=1
        apply_interval_dict['apply_interval_1_day'] += 1
    elif val > 1 and val <= 3:
        apply_interval_dict['apply_interval_1_3']+=1
        apply_interval_dict['apply_interval_3_day'] += 1
    elif val > 3 and val <= 7:
        apply_interval_dict['apply_interval_3_7']+=1
    elif val > 7 and val <= 14:
        apply_interval_dict['apply_interval_7_14']+=1
    elif val > 14 and val <= 30:
        apply_interval_dict['apply_interval_14_30']+=1
    elif val > 30 and val <= 60:
        apply_interval_dict['apply_interval_30_60']+=1
    elif val > 60 and val <= 90:
        apply_interval_dict['apply_interval_60_90']+=1
    elif val > 90 and val <= 180:
        apply_interval_dict['apply_interval_90_180']+=1
    elif val > 180 and val <= 360:
        apply_interval_dict['apply_interval_180_360']+=1
    elif val > 365:
        apply_interval_dict['apply_interval_360_9999']+=1

def split_record(dt, merchant_monthly, monthly, weekly, daily):
    """
    :param dt: original record data frame to list
    :param colname: time stamp
    :return: dict of all split record
    """

    merchant_monthly_dict = dict()
    monthly_dict = dict()
    weekly_dict = dict()
    daily_dict = dict()

    for row in dt:
        for i in loan_monthly:
            if i >= row[monthly]:
                if monthly_dict.get(i):
                    monthly_dict[i].append(row)
                else:
                    monthly_dict[i] = []
                    monthly_dict[i].append(row)

        for i in lh_week_map:
            if row[weekly] == i:
                if weekly_dict.get(i):
                    weekly_dict[i].append(row)
                else:
                    weekly_dict[i]=[]
                    weekly_dict[i].append(row)

        for i in  lh_day_map:
            if row[daily] == i:
                if daily_dict.get(i):
                    daily_dict[i].append(row)
                else:
                    daily_dict[i]=[]
                    daily_dict[i].append(row)

        for i in loan_merchant_monthly:
            if row[merchant_monthly] == i:
                if merchant_monthly_dict.get(i):
                    merchant_monthly_dict[i].append(row)
                else:
                    merchant_monthly_dict[i]=[]
                    merchant_monthly_dict[i].append(row)
    return monthly_dict,weekly_dict,daily_dict,merchant_monthly_dict

def split_record_by_type(dt, p_type):
    type_dict = dict()
    for row in dt:
        if row[p_type] == 2:
            if type_dict.get('int'):
                type_dict['int'].append(row)
            else:
                type_dict['int'] = []
                type_dict['int'].append(row)
        else:
            if type_dict.get('pdl'):
                type_dict['pdl'].append(row)
            else:
                type_dict['pdl'] = []
                type_dict['pdl'].append(row)

    return type_dict

def split_record_by_order(dt_list):
    order_repay_dict = dict()
    for re in dt_list:
        order_id = re[0]
        if order_repay_dict.get(order_id):
            order_repay_dict[order_id].append(re)
        else:
            order_repay_dict[order_id] = []
            order_repay_dict[order_id].append(re)

    return order_repay_dict

# 机构迁移指标计算
def loan_merchant_check(data):
    loan_merchant_feature = {}

    if data and len(data) > 0:
        curr_order = data.get(0)
        curr_pdl_count, curr_int_count, c_apr_pdl_count, c_apr_int_count = loop_loan_merchant(curr_order)
        # 总体机构申请通过情况
        sum_pdl_count, sum_int_count, sum_apr_pdl_count, sum_apr_int_count = curr_pdl_count, curr_int_count, c_apr_pdl_count, c_apr_int_count

        for key in range(1, 14):
            pre_order = data.get(key)
            pre_pdl_count, pre_int_count, p_apr_pdl_count, p_apr_int_count = loop_loan_merchant(pre_order)

            # 申请次数差异
            curr_pdl_diff = curr_pdl_count - pre_pdl_count
            curr_int_diff = curr_int_count - pre_int_count
            # 申请通过次数差异
            c_apr_pdl_diff = c_apr_pdl_count - p_apr_pdl_count
            c_apr_int_diff = c_apr_int_count - p_apr_int_count

            loan_merchant_feature['apply_mert_pdl_diff_' + str(key)] = curr_pdl_diff
            loan_merchant_feature['apply_mert_int_diff_' + str(key)] = curr_int_diff
            loan_merchant_feature['approve_mert_pdl_diff_' + str(key)] = c_apr_pdl_diff
            loan_merchant_feature['approve_mert_int_diff_' + str(key)] = c_apr_int_diff

            curr_order = pre_order
            curr_pdl_count, curr_int_count = pre_pdl_count, pre_int_count
            c_apr_pdl_count, c_apr_int_count = p_apr_pdl_count, p_apr_int_count

        loan_merchant_feature['apply_mert_pdl_sum'] = sum_pdl_count
        loan_merchant_feature['apply_mert_int_sum'] = sum_int_count
        loan_merchant_feature['approve_mert_pdl_sum'] = sum_apr_pdl_count
        loan_merchant_feature['approve_mert_int_sum'] = sum_apr_int_count


    return loan_merchant_feature

# 用户多头借贷行为指标
def loan_behavior_check(data):
    loan_behavior_feature = {}

    if len(data) > 0:
        curr_order = data.get(0)
        curr_pdl_count, curr_int_count,c_apr_pdl_count,c_apr_int_count = loop_loan_behavior(curr_order)

        for key in range(1,14):

            curr_diff = 0

            pre_order = data.get(key)
            pre_pdl_count, pre_int_count,p_apr_pdl_count,p_apr_int_count = loop_loan_behavior(pre_order)
            # print(pre_pdl_count, pre_int_count)

            # 申请次数差异
            curr_pdl_diff = curr_pdl_count - pre_pdl_count
            curr_int_diff = curr_int_count - pre_int_count
            # 申请通过次数差异
            c_apr_pdl_diff = c_apr_pdl_count - p_apr_pdl_count
            c_apr_int_diff = c_apr_int_count - p_apr_int_count

            loan_behavior_feature['apply_pdl_diff_'+str(key)] = curr_pdl_diff
            loan_behavior_feature['apply_int_diff_' + str(key)] = curr_int_diff
            loan_behavior_feature['approve_pdl_diff_' + str(key)] = c_apr_pdl_diff
            loan_behavior_feature['approve_int_diff_' + str(key)] = c_apr_int_diff

            curr_order = pre_order
            curr_pdl_count, curr_int_count = pre_pdl_count, pre_int_count
            c_apr_pdl_count, c_apr_int_count = p_apr_pdl_count,p_apr_int_count

    return loan_behavior_feature

# pdl int 计数
def loop_loan_behavior(orders):
    pdl_count,int_count = 0,0
    apr_pdl_count,apr_int_count = 0,0
    if  orders and len(orders):
        for prd in orders:
            loanNumber = prd[1]
            riskStatus = prd[2]
            # print(loanNumber)

            if loanNumber == '1':
                pdl_count += 1
                if riskStatus == 'Y':
                    apr_pdl_count += 1

            elif loanNumber == '2':
                int_count += 1
                if riskStatus == 'Y':
                    apr_int_count += 1

    return pdl_count, int_count, apr_pdl_count, apr_int_count


def merchant_feature_init():
    loan_merchant_feature = dict()
    for key in loan_merchant_monthly:
        loan_merchant_feature['apply_mert_pdl_diff_' + str(key)] = feature_no_order
        loan_merchant_feature['apply_mert_int_diff_' + str(key)] = feature_no_order
        loan_merchant_feature['approve_mert_pdl_diff_' + str(key)] = feature_no_order
        loan_merchant_feature['approve_mert_int_diff_' + str(key)] = feature_no_order
    loan_merchant_feature['apply_mert_pdl_sum'] = feature_no_order
    loan_merchant_feature['apply_mert_int_sum'] = feature_no_order
    loan_merchant_feature['approve_mert_pdl_sum'] = feature_no_order
    loan_merchant_feature['approve_mert_int_sum'] = feature_no_order

    return loan_merchant_feature

def behavior_feature_init():
    loan_merchant_feature = dict()
    for key in loan_merchant_monthly:
        loan_merchant_feature['apply_pdl_diff_' + str(key)] = feature_no_order
        loan_merchant_feature['apply_int_diff_' + str(key)] = feature_no_order
        loan_merchant_feature['approve_pdl_diff_' + str(key)] = 'NaN'
        loan_merchant_feature['approve_int_diff_' + str(key)] = feature_no_order


    return loan_merchant_feature

def loop_loan_merchant(orders):
    pdl_count, int_count = 0, 0
    apr_pdl_count, apr_int_count = 0, 0

    pdl_set,int_set,ap_pdl_set,ap_int_set = set(),set(),set(),set()
    if orders and len(orders):
        for prd in orders:
            loanNumber = prd[1]
            riskStatus = prd[2]
            merchantId = prd[3]
            # print(merchantId)

            if loanNumber == '1':
                pdl_set.add(merchantId)
                if riskStatus == 'Y':
                    ap_pdl_set.add(merchantId)

            elif loanNumber == '2':
                int_set.add(merchantId)
                if riskStatus == 'Y':
                    ap_int_set.add(merchantId)

    return len(pdl_set),len(int_set),len(ap_pdl_set),len(ap_int_set)

# 用户借贷机构迁移切片
def time_mert_loan_init(date_diff, order,date_dic):

    if date_diff >= 360:
        if date_dic.get(12) == None:
            date_dic[12] = []
            date_dic[12].append(order)
        else:
            date_dic[12].append(order)

    if date_diff >= 330:
        if date_dic.get(11) == None:
            date_dic[11] = []
            date_dic[11].append(order)
        else:
            date_dic[11].append(order)

    if date_diff >= 300:
        if date_dic.get(10) == None:
            date_dic[10] = []
            date_dic[10].append(order)
        else:
            date_dic[10].append(order)

    if date_diff >= 270:
        if date_dic.get(9) == None:
            date_dic[9] = []
            date_dic[9].append(order)
        else:
            date_dic[9].append(order)

    if date_diff >= 240:
        if date_dic.get(8) == None:
            date_dic[8] = []
            date_dic[8].append(order)
        else:
            date_dic[8].append(order)

    if date_diff >= 210:
        if date_dic.get(7) == None:
            date_dic[7] = []
            date_dic[7].append(order)
        else:
            date_dic[7].append(order)

    if date_diff >= 180:
        if date_dic.get(6) == None:
            date_dic[6] = []
            date_dic[6].append(order)
        else:
            date_dic[6].append(order)

    if date_diff >= 150:
        if date_dic.get(5) == None:
            date_dic[5] = []
            date_dic[5].append(order)
        else:
            date_dic[5].append(order)

    if date_diff >= 120:
        if date_dic.get(4) == None:
            date_dic[4] = []
            date_dic[4].append(order)
        else:
            date_dic[4].append(order)

    if date_diff >= 90:
        if date_dic.get(3) == None:
            date_dic[3] = []
            date_dic[3].append(order)
        else:
            date_dic[3].append(order)

    if date_diff >= 60:
        if date_dic.get(2) == None:
            date_dic[2] = []
            date_dic[2].append(order)
        else:
            date_dic[2].append(order)

    if date_diff >= 30:
        if date_dic.get(1) == None:
            date_dic[1] = []
            date_dic[1].append(order)
        else:
            date_dic[1].append(order)

    if date_dic.get(0) == None:
        date_dic[0] = []
        date_dic[0].append(order)
    else:
        date_dic[0].append(order)

#  用户借贷行为指标切片
def time_loan_init(date_diff, order, date_dic):

    if date_diff <= 30:
        if date_dic.get(1) == None:
            date_dic[1] = []
            date_dic[1].append(order)
        else:
            date_dic[1].append(order)

    if date_diff > 30 and date_diff <= 60:
        if date_dic.get(2) == None:
            date_dic[2] = []
            date_dic[2].append(order)
        else:
            date_dic[2].append(order)

    if date_diff > 60 and date_diff <= 90:
        if date_dic.get(3) == None:
            date_dic[3] = []
            date_dic[3].append(order)
        else:
            date_dic[3].append(order)

    if date_diff > 90 and date_diff <= 120:
        if date_dic.get(4) == None:
            date_dic[4] = []
            date_dic[4].append(order)
        else:
            date_dic[4].append(order)

    if date_diff > 120 and date_diff <= 150:
        if date_dic.get(5) == None:
            date_dic[5] = []
            date_dic[5].append(order)
        else:
            date_dic[5].append(order)

    if date_diff > 150 and date_diff <= 180:
        if date_dic.get(6) == None:
            date_dic[6] = []
            date_dic[6].append(order)
        else:
            date_dic[6].append(order)

    if date_diff > 180 and date_diff <= 210:
        if date_dic.get(7) == None:
            date_dic[7] = []
            date_dic[7].append(order)
        else:
            date_dic[7].append(order)

    if date_diff > 210 and date_diff <= 240:
        if date_dic.get(8) == None:
            date_dic[8] = []
            date_dic[8].append(order)
        else:
            date_dic[8].append(order)

    if date_diff > 240 and date_diff <= 270:
        if date_dic.get(9) == None:
            date_dic[9] = []
            date_dic[9].append(order)
        else:
            date_dic[9].append(order)

    if date_diff > 270 and date_diff <= 300:
        if date_dic.get(10) == None:
            date_dic[10] = []
            date_dic[10].append(order)
        else:
            date_dic[10].append(order)

    if date_diff > 300 and date_diff <= 330:
        if date_dic.get(11) == None:
            date_dic[11] = []
            date_dic[11].append(order)
        else:
            date_dic[11].append(order)

    if date_diff > 330:
        if date_dic.get(12) == None:
            date_dic[12] = []
            date_dic[12].append(order)
        else:
            date_dic[12].append(order)

# 计算日期差
def data_diff(sysDate, t_type,now):
    # 获取当前日期
    if now== -1:
        now = datetime.datetime.now()
    now = datetime.datetime.strftime(now, "%Y%m%d")

    date1 = time.strptime(now, "%Y%m%d")
    if t_type == 1:
        date2 = time.strptime(sysDate, "%Y%m%d")
    else:
        date2 = time.strptime(sysDate, "%Y-%m-%d %H:%M:%S")

    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])

    return date1 - date2

def addr_similar_check(address_all):

    same_addr_dict = dict()
    dis_addr_list = []

    # 字符相似度的阈值
    threadhold = 0.7
    # 1 过滤出相似度高于阈值的name
    print(address_all)

    for a in address_all:
        for b in address_all:
            if b != a:
                sim = difflib.SequenceMatcher(None, a, b).quick_ratio()
                if sim > threadhold:
                    if same_addr_dict.get(sim) == None:
                        dis_addr_list.append(a)
                    same_addr_dict[sim] = (a,b)


    print(same_addr_dict)
    print(dis_addr_list)
    return dis_addr_list

# 根据文本相似性去重
def addr_similar_check(address_all):

    same_addr_dict = dict()
    dis_addr_list = []
    temp = address_all[:]

    # 字符相似度的阈值
    threadhold = 0.7
    # 1 过滤出相似度高于阈值的name
    # print(address_all)

    for a in address_all:
        for b in temp:
            if b != a:
                sim = difflib.SequenceMatcher(None, a, b).quick_ratio()
                if sim > threadhold:
                    temp.remove(b)
                    address_all.remove(b)
                    same_addr_dict[sim] = (a,b)


    # print(same_addr_dict)
    # print(address_all)
    return address_all


