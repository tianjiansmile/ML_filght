#!-*- coding:utf8-*-
from com.risk_score.feature_extact import setting
import pymongo
import operator

# 对资源进行网络建模，主要用于网络数据准备

client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.resource_info


def build_node_rels():
    results = coll.find({}, {'_id': 0}).limit(100)

    node_dict = {}
    person_list = []
    device_list = []
    ip_list = []
    wifi_list = []
    imei_list = []

    device_dict = {}
    ip_dict = {}
    wifi_dict = {}
    imei_dict = {}
    for d in results:
        if d:
            for a in d:
                person_list.append(a)
                print(a)
                resource = d[a]
                deviceId = resource.get('deviceId')
                node_resource(a,deviceId,device_list)
                rel_resource(a,deviceId,device_dict)

                ip = resource.get('ip')
                node_resource(a,ip,ip_list)

                wifiIp = resource.get('wifiIp')
                node_resource(a, wifiIp, wifi_list)

                imei = resource.get('imei')
                node_resource(a, imei, imei_list)

    person_list = list(set(person_list))
    device_list = list(set(device_list))
    wifi_list = list(set(wifi_list))
    imei_list = list(set(imei_list))

# 遍历节点数据
def node_resource(idnum,deviceId,device_dict):
    for d in deviceId:
        if d and d.strip() != '':
            check = device_dict.get(d)
            if check == None:
                device_dict[d] = 1
            else:
                device_dict[d] += 1
            print(idnum,d)

def rel_resource(a,deviceId,device_list):
    pass




if __name__ == '__main__':
    # 构建网络所必须的node，rel文件
    build_node_rels()