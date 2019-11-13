from neo4j.v1 import GraphDatabase
import requests
from com.untils import idnum_handler

# 缺点是随着更新的数据，更新操作会越来越慢，可能和缓存有关
#  用GraphDatabase访问neo4j
class Neo4jHandler:
    # 对neo4j 进行读写
    def __init__(self,driver):
        self.driver = driver


    #     执行cypher语句
    def cypherexecuter(self, cypher):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(cypher)

                return result
        session.close()

    #     执行cypher语句with param
    def cypherexecuter_param(self, cypher,params):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(cypher,param=params)

                return result
        session.close()

import hashlib
def md5(src):
    m = hashlib.md5()
    m.update(str.encode(src))
    return m.hexdigest().lower()

# 对网络节点属性进行更新，主要是将节点的借贷历史情况更新到网络
def write_csv_neo4j(my_neo4j):

    # node 节点写入
    cypher_node = "match (p:person) where p.community=1477374 return p.nid as id"
    result = my_neo4j.cypherexecuter(cypher_node)
    for item in result:
        idNum = item[0]
        url = 'http://127.0.0.1:5000/getEncyUserFeaturesTest?identityNo=%s&currDate=20190418'
        res = requests.get(url % (idNum))
        if res.status_code == 200:
            all_list = []
            res = res.json()
            result = res.get('result')
            features = result.get('features')
            apply_sum_all = features.get('apply_sum_all')
            approve_sum_all = features.get('approve_sum_all')
            overdue_sum_all = features.get('overdue_sum_all')
            maxOverdue_sum_all = features.get('maxOverdue_sum_all')
            if maxOverdue_sum_all == -99999:
                maxOverdue_sum_all = 0

            update_cypher = "match (n:person) where n.nid='"+str(idNum)+"' set n.apply="+str(apply_sum_all)+" set n.approve="+str(approve_sum_all)\
                            +" set n.overdue="+str(overdue_sum_all)+" set n.maxoverdue="+str(maxOverdue_sum_all)+" return n"
            print(update_cypher)
            if apply_sum_all:
                result = my_neo4j.cypherexecuter(update_cypher)
                print(result)

#  读入网络节点数据 节点的借贷历史情况更新到网络
def his_loan_to_neo4j(my_neo4j):
    # 网络节点汇集
    ency_ids = {}
    # node 节点写入
    cypher_node = "match (p:person)  return p.nid as id, p.community as comm "
    result = my_neo4j.cypherexecuter(cypher_node)
    for item in result:
        idNum = item[0]
        comm = item[1]
        # print(idNum,age)
        ency_ids[idNum] = comm

    return ency_ids

def update_network(my_neo4j,history_dict,ency_ids):

    batch_size = 5000
    times = 0
    batch_list = []
    for his in history_dict:
        md5_id = md5(his)

        # 网络中存在
        if md5_id in ency_ids:
            if len(his) == 18:  # 正常身份证
                try:
                    age, gender = get_age(his)
                except:
                    age = -1
                    gender = -1
            else:
                age = -1
                gender = -1
            loan = history_dict[his]

            temp = {}
            temp['nid'] = md5_id
            temp['sex'] = gender
            temp['age'] = age
            temp['apply'] = loan[0]
            temp['approve'] = loan[1]
            temp['overdue'] = loan[2]
            temp['loanamount'] = loan[3]
            if loan[4] == -99999:
                max_overdue = -1
            else:
                max_overdue = loan[4]
            temp['max_overdue'] = max_overdue

            batch_list.append(temp)

        if len(batch_list) == batch_size:
            batch_update(my_neo4j, batch_list,batch_size)
            batch_list = []
            times+=1
            print(times)
            # break

    print('最后一次update---------------------------------------')
    batch_update(my_neo4j, batch_list, batch_size)

#  将所有更新的节点重新写成node文件
def update_network_to_file(my_neo4j,history_dict,ency_ids):

    times = 0
    node = {}
    for item in history_dict:
        id = item
        md5_id = md5(id)

        # 网络中存在
        check = ency_ids.get(md5_id)
        if check:
            if len(id) == 18:  # 正常身份证
                try:
                    age, gender = get_age(id)
                except:
                    age = -1
                    gender = -1
            else:
                age = -1
                gender = -1
            loan = history_dict[item]

            if loan[4] == -99999:
                max_overdue = -1
            else:
                max_overdue = loan[4]

            node[md5_id] = (check,age,gender,loan[0],loan[1],loan[2],loan[3],max_overdue)


        # times+=1
        # print(times)

    return node

# 原网络中的数据必须全部写入文件，无论是否拿到历史特征数据
def write_node_file(ency_ids,node):
    count = 0
    with open('D:/develop/data/network/md5node_loan.csv', 'w') as nwf:
        nwf.write('nid:ID,community,age,sex,apply,approve,overdue,loanamount,max_overdue,is_black,:LABEL' + '\n')
        for item in ency_ids:
            if node.get(item):
                loan = node.get(item)
                nwf.write(item + ',' + str(loan[0]) + ',' + str(loan[1]) + ',' + str(loan[2]) + ',' + str(
                    loan[3]) + ',' + str(loan[4]) + ',' + str(loan[5]) + ',' + str(loan[6]) + ',' + str(
                    loan[7]) + ',0,person' + '\n')
                count+=1
            else:
                nwf.write(item + ','+str(ency_ids.get(item))+',-1,-1,-1,-1,-1,-1,-1,0,person' + '\n')

    print('updated:',count)

# 批量更新节点属性
def batch_update(my_neo4j,batch_list,batch_size):
    if len(batch_list) >0:
        # 构造
        batch = '['
        count = 0
        for dict_item in batch_list:
            count+=1
            batch += '{'
            for item in dict_item:
                if item =='nid':
                    batch+=item+":'"+dict_item[item]+"'"
                else:
                    batch += item + ":" + str(dict_item[item])

                if item != 'max_overdue':
                    batch +=','

            if count==len(batch_list):
                batch += '}'
            else:
                batch += '},'
        batch += ']'
        # 执行更新操作
        merge = "UNWIND "+batch+" as row MERGE (n:person {nid:row.nid}) " \
                                 "SET n.sex = row.sex, n.age = row.age," \
                                 "n.apply = row.apply, n.approve = row.approve," \
                                 "n.overdue = row.overdue, n.loanamount = row.loanamount," \
                                 "n.max_overdue = row.max_overdue"

        result = my_neo4j.cypherexecuter(merge)
        # print(batch)
        # print(result)

# 获取性别和年龄
def get_age(idnum):
    age = idnum_handler.GetInformation(idnum).get_age()
    gender = idnum_handler.GetInformation(idnum).get_sex()
    return age,gender

# 对neo4j批量更新进行测试
def batch_update_test(my_neo4j):
    # 构造一个list
    batch = "[{nid:'6f94f0f9a9d5d464f81cfff5c7014b96',sex:1,age:32,apply:3,approve:2,overdue:1,loanamount:2000.00,max_overdue:10}," \
                     "{nid:'9ccdbfecef0cd07632498a82569a628c',sex:1,age:22,apply:3,approve:2,overdue:1,loanamount:2000.00,max_overdue:10}]"
    # 执行更新操作
    merge = "UNWIND "+batch+" as row MERGE (n:person {nid:row.nid}) " \
                             "SET n.sex = row.sex, n.age = row.age," \
                             "n.apply = row.apply, n.approve = row.approve," \
                             "n.overdue = row.overdue, n.loanamount = row.loanamount," \
                             "n.max_overdue = row.max_overdue"

    print(merge)

    # my_neo4j.cypherexecuter_param(merge,batch)
    #
    #
    # my_neo4j.cypherexecuter_param(merge,batch)
if __name__ == "__main__":
    from com.NLP.jieba import jieba_test
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

    my_neo4j = Neo4jHandler(driver)

    batch_update_test(my_neo4j)

    # write_csv_neo4j(my_neo4j)

    # 1从文件读入历史借贷表现数据
    # history_dict = jieba_test.order_dict()
    # print('读入历史数据完毕',len(history_dict))

    # 2 读入网络节点
    # ency_ids = his_loan_to_neo4j(my_neo4j)
    # print('读入网络数据完毕', len(ency_ids))

    # 3 开始更新
    # update_network(my_neo4j,history_dict,ency_ids)


    # 将更新节点重新写入
    # print('开始撞数据')
    # node = update_network_to_file(my_neo4j,history_dict,ency_ids)
    # print('开始更新')
    # write_node_file(ency_ids, node)
    # print(my_neo4j)
