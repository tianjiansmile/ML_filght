#!-*- coding:utf8-*-
from com.risk_score.feature_extact import setting
import pymongo
import operator

# 对资源进行网络建模，主要用于网络数据准备

client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.resource_info


def build_node_rels():
    # results = coll.find({}, {'_id': 0}).limit(20000)
    results = coll.find({}, {'_id': 0})


    node_dict = {}
    # 存放node实体
    person_list = []
    device_list = []
    ip_list = []
    wifi_list = []
    imei_list = []

    # 存放关系数据
    rel_list = []
    for d in results:
        if d:
            for a in d:
                person_list.append(a)
                print(a)
                resource = d[a]
                deviceId = resource.get('deviceId')
                node_resource(a,deviceId,device_list)
                rel_resource(a,deviceId,rel_list,'device')

                ip = resource.get('ip')
                node_resource(a,ip,ip_list)
                rel_resource(a, ip, rel_list, 'ip')

                wifiIp = resource.get('wifiIp')
                # node_resource(a, wifiIp, wifi_list)
                # rel_resource(a, wifiIp, rel_list,'wifiip')

                imei = resource.get('imei')
                # node_resource(a, imei, imei_list)
                # rel_resource(a, imei, rel_list,'imei')

    print('person_list去重前', len(person_list))
    person_list = list(set(person_list))
    print('person_list去重后', len(person_list))

    print('device_list去重前',len(device_list))
    device_list = list(set(device_list))
    print('device_list去重后', len(device_list))

    print('ip_list去重前', len(ip_list))
    ip_list = list(set(ip_list))
    print('ip_list去重后', len(ip_list))

    # print('wifi_list去重前', len(wifi_list))
    # wifi_list = list(set(wifi_list))
    # print('wifi_list去重后', len(wifi_list))
    #
    # print('imei_list去重前', len(imei_list))
    # imei_list = list(set(imei_list))
    # print('imei_list去重后', len(imei_list))

    # print(rel_list)

    node_to_file(person_list,device_list,ip_list,imei_list)
    rel_to_file(rel_list)

def node_to_file(person_list,device_list,ip_list,imei_list):
    with open('D:/develop/data/network/resource_data/md5node.csv', 'w') as nwf:
        nwf.write('nid:ID,is_black,overdue,:LABEL' + '\n')
        for re in person_list:
            md5_id = md5(re)
            nwf.write(md5_id + ',-1,0,person' + '\n')

        for re in device_list:
            nwf.write(re + ',-1,0,device' + '\n')

        for re in ip_list:
            nwf.write(re + ',-1,0,ip' + '\n')

        for re in imei_list:
            nwf.write(re + ',-1,0,imei' + '\n')

# rels.csv 格式
# :START_ID,:END_ID,:TYPE
def rel_to_file(rel_list):
    with open('D:/develop/data/network/resource_data/md5rels.csv', 'w') as nwf:
        nwf.write(':START_ID,:END_ID,:TYPE' + '\n')
        for re in rel_list:
            md5_id = md5(re[0])
            nwf.write(md5_id + ',' + re[2] + ',' + re[1]+ '\n')

import hashlib
def md5(src):
    m = hashlib.md5()
    m.update(str.encode(src))
    return m.hexdigest().lower()

# 遍历节点数据
def node_resource(idnum,deviceId,device_dict):
    # print(idnum,deviceId,device_dict)

    for d in deviceId:
        # print(d)
        if d and d.strip() != '':
            if ',' not in d:
                device_dict.append(d)

# 关系
def rel_resource(a,deviceId,rel_list,mark):
    for d in deviceId:
        if d and d.strip() != '':
            if ',' not in d:
                temp = (a,mark,d)
                rel_list.append(temp)





if __name__ == '__main__':
    # 构建网络所必须的node，rel文件
    build_node_rels()