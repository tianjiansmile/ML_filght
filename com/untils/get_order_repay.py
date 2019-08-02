import pymysql
from com.risk_score.feature_extact import setting
import pymongo
import datetime
import time

# 获取贷后标签数据，可以获取，pld int 或是指定机构的贷后数据

def getJinpanOrders(data_index,loan_number,merchant_id,sys_date,wf):
    read_con = pymysql.connect(host=setting.mysql_host, user=setting.mysql_user,
                               password=setting.mysql_pass, database=setting.musql_db + data_index, port=3306,
                               charset='utf8')

    cursor = read_con.cursor()
    sql = "select order_id,repay_state,repay_date,actual_date from user_repayment_record " \
          "where sys_date = %s and loan_number=%s "

    # loan_sql = "and loan_number='%d' "
    merchant_sql = "and merchant_id=%s "
    if merchant_id:
        sql = sql + merchant_sql
        cursor.execute(sql,(sys_date,int(loan_number),int(merchant_id)))
    else:
        cursor.execute(sql, (sys_date, int(loan_number)))

    result = cursor.fetchall()
    for re in result:
        print(re)
        order_id = re[0]
        repay_state = re[1]
        repay_date = re[2]
        actual_date = re[3]

        overdureday = check_overdueday(repay_state,repay_date,actual_date)
        wf.write(order_id+','+str(overdureday)+'\n')

def check_overdueday(repay_state,repay_date,actual_date):

    overdureday = None
    # 获取当前日期
    now_date = datetime.datetime.now().date()

    repays_date = repay_date.date()

    if repay_state == '2': #还款标志
        if actual_date:

            actuals_date = actual_date.date()
            overdureday = (actuals_date - repays_date).days
            if overdureday == 1:
                overduretime = actual_date - repay_date
                if overduretime.seconds < 86400: # 小于一天
                    overdureday = 0


        else:
            overdureday = (now_date - repays_date).days
            if overdureday == 1:
                overduretime = datetime.datetime.now() - repay_date
                if overduretime.seconds < 86400: # 小于一天
                    overdureday = 0
    else:
        overdureday = (now_date - repays_date).days
        if overdureday == 1:
            overduretime = datetime.datetime.now() - repay_date
            if overduretime.seconds < 86400:  # 小于一天
                overdureday = 0

    if overdureday < 0:
        overdureday = 0
    return overdureday

def check_overdueday(repay_state,repay_date,actual_date,merchant_id):

    overdureday = None
    niwodai = 4

    # 获取当前日期
    now_date = datetime.datetime.now().date()

    repays_date = repay_date.date()

    if repay_state == '2': #还款标志
        if actual_date:

            actuals_date = actual_date.date()
            overdureday = (actuals_date - repays_date).days
            if overdureday == 1:
                if merchant_id == '11':
                    hour = actual_date.hour
                    if hour < niwodai:
                        overdureday = 0
                    print('你我贷：', hour)


        else:
            overdureday = (now_date - repays_date).days
            if overdureday == 1:
                if merchant_id == '11':
                    hour = now_date.hour
                    if hour < niwodai:
                        overdureday = 0
                    print('你我贷：', hour)
    else:
        overdureday = (now_date - repays_date).days
        if overdureday == 1:
            if merchant_id == '11':
                hour = now_date.hour
                if hour < niwodai:
                    overdureday = 0
                print('你我贷：',hour)

    # print('overdureday',overdureday)
    if overdureday < 0:
        overdureday = 0
    return overdureday

# loan_number 分期orpdl merchant_id 机构号
def get_order_label(loan_number=1,merchant_id=None):
    # 这个时间组件不能跨月遍历
    days = getDays('20190501','20190520')

    with open('data/'+merchant_id+'_pdl_label.csv','w') as wf:
        wf.write('order_id,overdueday'+'\n')
        for sys_date in days:
            print(sys_date)
            for i in range(10):
                getJinpanOrders(str(i),loan_number,merchant_id,sys_date, wf)

def getDays(start, end):
    days = []
    begin = datetime.datetime.strptime(start, '%Y%m%d')
    end = datetime.datetime.strptime(end, '%Y%m%d')
    for i in range((end - begin).days + 1):
        day = begin + datetime.timedelta(days=i)
        days.append(day.strftime('%Y%m%d'))
    return tuple(days)

def getAddr(order_id):
    pass


if __name__ == '__main__':
    # 1 获取通过的订单号对应的逾期天数 ,同信云盾数据
    get_order_label(loan_number=1,merchant_id='3410')

    # 2 获取该订单的地址