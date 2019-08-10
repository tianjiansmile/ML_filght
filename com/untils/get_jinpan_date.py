import pymysql
from com.risk_score.feature_extact import setting
import pymongo
import datetime
import pandas as pd
import time

# 从数据库获取订单对应的地址数据
# 地址数据可以衍生的特征主要有
# 1 统计数据： 全量借贷历史数据指标映射至地址，形成地址映射表，对新地址进行解析并获得该地址下的历史表现指标
# 2 模糊匹配： 一些关键词的搜索，

read_con = pymysql.connect(host=setting.jin_host, user=setting.jin_user,
                               password=setting.jin_pass, database=setting.jin_db, port=8066,
                               charset='utf8')

cursor = read_con.cursor()

def get_addr_data(order_id):

    sql = "select addr_detail from user_address_info where addr_type = 'L' and order_id=%s "
    addr = None
    try:
        cursor.execute(sql, (order_id))
        result = cursor.fetchone()
        if result:
            for addr in result:
                print(order_id,addr)
        else:
            print('fuck')

    except:
        print('error')

    return addr





if __name__ == '__main__':
    import time

    starttime = time.time()

    merchant_id = '3410'
    data = pd.read_csv('data/'+merchant_id+'_pdl_label.csv')
    order_ids = data['order_id']

    print('读入数据完成')
    # get_addr_data(order_ids)

    data['addr'] = data['order_id'].map(get_addr_data)

    data.to_csv('data/'+merchant_id+'addr_label.csv')

    cursor.close()
    read_con.close()
    endtime = time.time()
    print(' cost time: ', endtime - starttime)
