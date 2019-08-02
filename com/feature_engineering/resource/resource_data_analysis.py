import pymysql
from com.risk_score.feature_extact import setting
import pandas as pd
import json
import datetime
from com.untils import loan_history_utils as lhu
import numpy as np

# 本脚本意在通过对逾期用户和非逾期用户的历史资源数据的观察
# 分析来探索挖掘出有效的历史数据特征,并写入文件


# not used for now
class JinPanData():
    def __init__(self, data_configs, is_live=False):

        self.data_configs = data_configs

    def connection_jinpan(self):
        return pymysql.connect(host=self.data_configs.tidb_host,
                                                 user=self.data_configs.tidb_user,
                                                 passwd=self.data_configs.tidb_pwd,
                                                 port=3306,
                                                 db=self.data_configs.tidb_db,
                                                 charset='utf8')

    def connection_mycat(self):
        return pymysql.connect(host=self.data_configs.mycat_host,
                                                 user=self.data_configs.mycat_user,
                                                 passwd=self.data_configs.mycat_pwd,
                                                 port=8066,
                                                 db=self.data_configs.mycat_db,
                                                 charset='utf8')



    def get_df_order(self,identity_no,conn):
        sql = 'select uo.order_id,uo.loan_number,uo.risk_status,uo.merchant_id,uo.model_data,uo.sys_date,uo.create_time' \
              ' from user_order uo left join user_basics_info ub on uo.user_id = ub.user_id ' \
              "where ub.identity_no_ency= '{identity_no}'" \
              ' order by uo.sys_date desc'.format(
            identity_no=identity_no)

        columns = ['order_id','loan_number', 'risk_status', 'merchant_id','model_data', 'sys_date','create_time']
        # dt = self.convert_to_df(columns, r)
        dt = self.convert_to_df_pro(sql,conn)


        return dt

    def get_order_user(self, order_id, conn):
        sql = 'select uo.order_id,ub.identity_no from user_order uo left join user_basics_info ub  on uo.user_id = ub.user_id ' \
              " where uo.order_id = '{order_id}'".format(
            order_id=order_id)

        r = self.sloop_execute_sql(conn, sql)

        return r

    def get_df_batch_order(self,identity_nos,conn):
        sql = 'select ub.identity_no_ency,uo.order_id,uo.loan_number,uo.risk_status,uo.merchant_id,uo.model_data,uo.sys_date,uo.create_time' \
              ' from user_order uo left join user_basics_info ub on uo.user_id = ub.user_id ' \
              "where ub.identity_no_ency in {identity_nos}" \
              ' '.format(
            identity_nos=identity_nos)
        dt = self.convert_to_df_pro(sql, conn)

        return dt

    def get_df_repay(self,identity_no,conn):
        sql = 'select ur.order_id,uo.merchant_id,ur.loan_number,ur.repay_date,ur.actual_date,ur.repay_amount,ur.repay_state,ur.settle_time ' \
              'from user_repayment_record ur left join user_basics_info ub on ur.user_id = ub.user_id ' \
              'left join user_order uo on uo.user_id = ur.user_id and uo.order_id = ur.order_id ' \
              "where ub.identity_no_ency= '{identity_no}'" \
              ' order by ur.repay_date desc'.format(
            identity_no=identity_no)

        dt = self.convert_to_df_pro(sql, conn)
        return dt

    def get_df_batch_repay(self,identity_nos,conn):
        sql = 'select ub.identity_no_ency,ur.order_id,uo.merchant_id,ur.loan_number,ur.repay_date,ur.actual_date,ur.repay_amount,ur.repay_state,ur.settle_time ' \
              'from user_repayment_record ur left join user_basics_info ub on ur.user_id = ub.user_id ' \
              'left join user_order uo on uo.user_id = ur.user_id and uo.order_id = ur.order_id ' \
              "where ub.identity_no_ency in {identity_nos}" \
              ' '.format(
            identity_nos=identity_nos)
        dt = self.convert_to_df_pro(sql, conn)
        return dt

    def get_df_resources(self,identity_no):
        sql = 'select distinct uo.order_id,uo.merchant_id,uo.loan_number,ua.sys_date,ua.addr_detail, ' \
              'uc.compy_name,uc.compy_addr,ud.model,ud.device_no,ucr.user_data,ucr.user_phone ' \
              'from user_order uo left join user_basics_info ub on uo.user_id = ub.user_id ' \
              'left join user_address_info ua on uo.order_id = ua.order_id and uo.user_id = ua.user_id ' \
              'left join user_company_info uc on uo.order_id = uc.order_id and uo.user_id = uc.user_id ' \
              'left join user_device ud on uo.order_id = ud.order_id and uo.user_id = ud.user_id ' \
              'left join user_contact_info ucr on uo.order_id = ucr.order_id and uo.user_id = ucr.user_id ' \
              "where ub.identity_no= '{identity_no}'" \
              " and uo.merchant_id !='50' and ucr.type = '2' " \
              " and ua.addr_type = 'L' order by ua.sys_date desc".format(
            identity_no=identity_no)

        import time
        starttime = time.time()
        conn = self.connection_mycat()
        dt = self.convert_to_df_pro(sql, conn)
        endtime = time.time()
        print(' resource 查询cost time: ', endtime - starttime)
        return dt

    @staticmethod
    def convert_to_df_pro(sql, connection):
        return pd.read_sql(sql, connection)

    @staticmethod
    def sloop_execute_sql(connection, sql):
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchone()
            return result
        # todo too general
        except Exception as e:
            print(e)
            return None


class LoanHistoryResourceFeatureGroup( JinPanData ):
    def __init__(self, data_configs, is_live=False):
        super(LoanHistoryResourceFeatureGroup, self).__init__(data_configs, is_live)

        self.reject_overdue = -3

        self.default_val = 0

        self.type_dict = dict()

        self.resource_fea = ['number','dis_number','dis_rate','real_number','real_rate']

        # 特征池
        self.feature_pool = dict()

        self.merchant_order_dict = dict()

        self.emergence_dict = dict()

    # 数据分片
    def type_order_data_build(self, dt, timeback):
        fea = dict()
        if timeback == -1:
            cut_time = datetime.datetime.now()
            dt = lhu.apply_type_window(dt, 'loan_number')
        else:
            pass

        # print(dt[['merchant_id','p_type','addr_detail']])
        # print(dt[['merchant_id', 'p_type', 'compy_name', 'compy_addr']])
        # print(dt[['order_id', 'merchant_id', 'sys_date', 'device_no']])
        # print(dt[['order_id','merchant_id', 'p_type','user_phone']])
        # print(dt[['merchant_id', 'p_type', 'user_data']])

        dt_list = dt.values.tolist()
        type_dict = lhu.split_record_by_type(dt_list, 2)
        self.type_dict = type_dict

        for t in type_dict:
            intm = type_dict.get('int')
            pdl = type_dict.get('pdl')



        # 居住地址
        addr_list = dt['addr_detail']
        live_addr_fea = self.resource_loop(addr_list.tolist(),'live_addr')
        # 公司名称
        comp_list = dt['compy_name']
        comp_name_fea = self.resource_loop(comp_list.tolist(), 'compy_name')
        # 公司地址
        compy_addr = dt['compy_addr']
        comp_addr_fea = self.resource_loop(compy_addr.tolist(), 'compy_addr')
        # 设备信息
        device_list = dt['device_no']
        device_fea = self.device_loop(device_list.tolist(), 'device_no')

        # 电话
        user_phone = dt['user_phone']

        emerge_contact = dt['user_data']
        # print(emerge_contact)
        emergence_fea = self.emergence_loop(emerge_contact)

        fea.update(live_addr_fea)
        fea.update(comp_name_fea)
        fea.update(comp_addr_fea)
        fea.update(device_fea)
        fea.update(emergence_fea)

        self.feature_pool = fea


    def emergence_loop(self,emerge_contact):
        fea_dict = dict()

        phone_name = dict()
        phone_relation = dict()
        for emer in emerge_contact:
            emer = json.loads(emer)
            print(emer)
            for item in emer:
                phone = item.get('phone')
                phone = phone.replace(' ','')
                relation = item.get('relation')
                name = item.get('name')

                check = phone_name.get(phone)
                if name:
                    if check:
                        name = name.replace(' ','')
                        check.add(name)
                    else:
                        phone_name[phone] = set()
                        phone_name[phone].add(name)

                if relation:
                    relation = relation.replace(' ', '')
                    check = phone_relation.get(phone)
                    if check:
                        check.add(relation)
                    else:
                        phone_relation[phone] = set()
                        phone_relation[phone].add(relation)

                    rel_name_dict.add(relation)

        print(phone_name)
        print(phone_relation)

        phone_name_number = [len(i[1]) for i in phone_name.items()]
        phone_relation_number = [len(i[1]) for i in phone_relation.items()]

        phone_count = len(phone_name)
        mean_phone_name_number = round(float(np.mean(phone_name_number)), 2)
        std_phone_name_number = round(float(np.std(phone_name_number,ddof=1)), 2)
        max_phone_name_number = np.max(phone_name_number)
        counts = np.bincount(phone_name_number)
        # 返回众数
        mode_phone_name_number = np.argmax(counts)

        mean_phone_relation_number = round(float(np.mean(phone_relation_number)), 2)
        std_phone_relation_number = round(float(np.std(phone_relation_number, ddof=1)), 2)
        max_phone_relation_number = np.max(phone_relation_number)
        counts = np.bincount(phone_relation_number)
        # 返回众数
        mode_phone_relation_number = np.argmax(counts)

        print('name count', phone_name_number,'fea',mean_phone_name_number,std_phone_name_number,max_phone_name_number,mode_phone_name_number)
        print('relation count', phone_relation_number,'fea',mean_phone_relation_number,std_phone_relation_number,max_phone_relation_number,mode_phone_relation_number)

        fea_dict = {'emer_phone_count':phone_count,'mean_phone_name_number':mean_phone_name_number,
        'std_phone_name_number':std_phone_name_number,'max_phone_name_number':max_phone_name_number,
        'mode_phone_name_number':mode_phone_name_number,'mean_phone_relation_number':mean_phone_relation_number,
        'std_phone_relation_number':std_phone_relation_number,'max_phone_relation_number':max_phone_relation_number,
        'mode_phone_relation_number':mode_phone_relation_number}

        return fea_dict

    def resource_loop(self,resource,column):
        fea = []
        fea_dict = dict()
        # 去空
        resource = [i for i in resource if i]
        # print(resource)
        resource_len = len(resource)
        # 去重
        dis_res = set(resource)
        dis_res_len = len(dis_res)

        # 再一次去重
        if dis_res_len > 1:
            real_res = lhu.addr_similar_check(list(dis_res))
        else:
            real_res = dis_res

        real_res_len = len(real_res)

        print(resource_len, dis_res_len,real_res_len)
        fea.append(resource_len)
        fea.append(dis_res_len)

        if resource_len != 0:
            dis_len_rate = round(float(dis_res_len / resource_len),2)
            fea.append(dis_len_rate)
        else:
            fea.append(0)

        fea.append(real_res_len)

        if dis_res_len != 0:
            real_len_rate = round(float(real_res_len / dis_res_len),2)
            fea.append(real_len_rate)
        else:
            fea.append(0)


        for i in range(len(self.resource_fea)):


            col = 'his_' + column + '_' + self.resource_fea[i]
            val = fea[i]
            fea_dict[col] = val

        print(fea_dict)
        return fea_dict


    def device_loop(self,resource,column):
        fea = []
        fea_dict = dict()
        # 去空
        resource = [i for i in resource if i]
        # print(resource)
        resource_len = len(resource)
        # 去重
        dis_res = set(resource)
        dis_res_len = len(dis_res)

        fea.append(resource_len)
        fea.append(dis_res_len)

        if resource_len != 0:
            dis_len_rate = round(float(dis_res_len / resource_len),2)
            fea.append(dis_len_rate)
        else:
            fea.append(0)


        for i in range(len(self.resource_fea)-2):


            col = 'his_' + column + '_' + self.resource_fea[i]
            val = fea[i]
            fea_dict[col] = val

        print('device', fea_dict)
        return fea_dict



    def _feat_loan_history(self, identity_no, timeback):


        # 地址迁移指标
        resource_dt = self.get_df_resources(identity_no)

        self.type_order_data_build(resource_dt,timeback)

        # self.feature_pool.update(self.merchant_order_feature)


    def calc_the_group(self, identity_no,timeback = -1):
        # join with feature json and if not calculated, replace by missing value.
        all_feat_dict = dict()
        self._feat_loan_history(identity_no,timeback)
        return all_feat_dict

def res_check(file,loan,write_file):
    overdue_order = []
    conn = loan.connection_jinpan()
    with open(file,'r',encoding='utf8') as rf:
        with open(write_file, 'w',encoding='utf8') as wf:
            for line in rf.readlines()[1:5000]:
                line = line.replace('\n','')
                temp = line.split(',')
                # print(temp)
                order_id = temp[0]
                overdueday = temp[1]
                try:
                    wf.write(order_id+','+overdueday+',')
                    overdue_order.append(order_id)
                    feature_pool = get_resource(order_id, loan,conn)
                    temp_str = ''
                    for fea in feature_pool:
                        temp_str+=str(feature_pool[fea])
                        temp_str += ','

                    temp_str = temp_str[:-1]
                    wf.write(temp_str)
                    wf.write('\n')
                    # if int(overdueday) == 0:
                    #     overdue_order.append(order_id)
                    #     get_resource(order_id, loan,conn)
                except:
                    conn = loan.connection_mycat()
                    wf.write('\n')

            # if len(overdue_order) > 1000:
            #     break

def get_resource(order_id,loan,conn):
    result = loan.get_order_user(order_id,conn)
    identity_no = result[1]
    print(identity_no)
    # if identity_no != '320684199202213692':
    loan.calc_the_group(identity_no)
    return loan.feature_pool



if __name__ == '__main__':
    import time
    from com.risk_score.feature_extact import setting
    starttime = time.time()
    loan = LoanHistoryResourceFeatureGroup(setting)
    identity_no = '321324198307105413'  # 有分期贷后
    # identity_no = '62010219868227531X'   # 无订单
    # identity_no = '440881199005274137' # 一个订单有多个地址
    # identity_no = '370302199503267713'
    # identity_no = '110101197303275319'
    # identity_no = '110101197304072011'
    identity_no = '350582199201083555'  # 超多订单
    # identity_no = '232303199806056216'
    # identity_no = '452629198403070925'
    # loan.calc_the_group(identity_no)
    # print(json.dumps(loan.feature_pool))
    # print(len(loan.feature_pool))

    rel_name_dict = set()
    file = 'data/3410addr_label_201905.csv'
    write_file = 'resource_fea_label.csv'
    res_check(file,loan,write_file)
    endtime = time.time()

    print(rel_name_dict)
    print('cost time: ', endtime - starttime)