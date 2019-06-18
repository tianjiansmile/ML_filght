# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
from neo4j.v1 import GraphDatabase
from com.untils import public_function
import json

# 通过这个文件实现将neo4j的子图转换为networkx可以读入的数据文件

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

#  用GraphDatabase访问neo4j
class Neo4jHandler:
    # 对neo4j 进行读写
    def __init__(self,driver):
        self.driver = driver

    # 查出所有数据，并已列表返回
    def listreader(self, cypher, keys='tom'):

        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                data = []
                person = keys[0];movie = keys[1]
                result = tx.run(cypher)
                for record in result:
                    # 查看一个未知对象的类型  查询到的是对象和对象的嵌套， <Record: <Node: person>, <Node:movie>>
                    # type(record)
                    # 查看一个对象的所有方法
                    p_id = record[person].id
                    m_id = record[movie].id

                    p = dict(record[person])
                    p['p_id'] = p_id
                    m = dict(record[movie])
                    m['m_id'] = m_id
                    # 合并两个字典
                    p_m = dict(p,**m)
                    data.append(p_m)

                return data

        session.close()


    #     执行cypher语句
    def cypherexecuter(self, cypher):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(cypher)
                return result
        session.close()

# 将neo4j子图转化为文件
def neo4j_to_file(filename,data):
    with open(filename,'w') as wf:
        for record in data:
            print(record[0], record[1], record[2], record[3], record[4])
            sid = record[0]
            tid = record[1]
            rel = record[2]
            call_len = record[3]
            times = record[4]

            if rel == 'contact':
                call_len = '2'
                times = '2'
            if rel == 'same_phone':
                call_len = '999'
                times = '999'

            wf.write(str(sid)+'\t'+str(tid)+'\t'+rel+'\t'+call_len+'\t'+times+'\n')

# 对网络中的社区进行简单统计分析
def comm_count(my_neo4j):
    cypher_read = "MATCH (u:person) RETURN u.partition as partition,count(*) as size_of_partition ORDER by partition DESC"
    data = my_neo4j.cypherexecuter(cypher_read)
    count = 0
    c_10 = 0
    c_100 = 0
    for da in data:
        count+=1
        if da[1] >10:
            c_10+=1
            if da[1] > 100:
                c_100+=1
        else:
            print(da)

    print(count,'c_10',c_10,'c_100',c_100)
if __name__ == '__main__':
    my_neo4j = Neo4jHandler(driver)
    # print(my_neo4j)
    comm = '8998331'

    cypher_read = "match path = (p:person)-[a]-(q:person) where p.community='" + comm + "' and q.community='" + comm + "' " \
                   " return p.nid as sid,q.nid as tid,type(a) as rel,a.call_len as " \
                   " call_len, a.time as times"
    # data = my_neo4j.cypherexecuter(cypher_read)
    #
    # filename = 'community_detection/data/'+comm+'edgelist.txt'
    # neo4j_to_file(filename, data)

    # 对网络中的社区进行简单统计分析
    # comm_count(my_neo4j)

    # 尝试对网络中的超级大分区进行社团化
    # part=2516272
    # cypher_read = "match path = (p:person)-[a]-(q:person) where p.partition=" + comm + " and q.partition=" + comm + " " \
    #                " return p.nid as sid,q.nid as tid,type(a) as rel,a.call_len as " \
    #                " call_len, a.time as times"
    #
    # filename = 'community_detection/data/'+comm+'edgelist.txt'
    # neo4j_to_file(filename, data)