
#!-*- coding:utf8-*-
import pymongo
import jieba
import cpca
from com.NLP.jieba import prov_dict
import operator;
from com.risk_score.feature_extact import setting
import json
import pymysql


# 目的是对身份证地址进行解析，得到 省市区镇村五个级别的具体地址
# 然后将用户历史借贷情况映射至身份证地址数据

client = pymongo.MongoClient(setting.host)
db = client.call_test
coll = db.call_info

def mongo_read():
    results = coll.find({}, {'_id': 0, 'emergencer': 0,'calls': 0,'contacts': 0,'phone': 0,'carr_phone': 0 }).limit(100)
    for d in results:
        if d:
            idnum = d.get('id_num')
            addresses = d.get('addresses')
            l_a = addresses.get('L')
            if l_a:
                temp = l_a[0]
                temp = temp.replace(',', '')
                temp = temp.replace('|', '')
                temp = temp.replace('-', '')
                temp = temp.replace(' ', '')
                temp = temp.strip()
                df = cpca.transform([temp])
                print(temp,df)

def mysql_read():
    read_con = pymysql.connect(host='rr-uf6v8tdgdp0a488baeo.mysql.rds.aliyuncs.com', user='sloop_reader',
                               password='CgFww9sOSskN3YYC', database='basics_user_' + str(0), port=3306,
                               charset='utf8')
    sql = 'select id_address from user_identity_info limit 1000'

    cur = read_con.cursor()
    cur.execute(sql)
    result = cur.fetchall()
    for r in result:
        try:
            temp = r[0]
            df = cpca.transform([temp])
            print(temp)
            prov = df['省'][0]
            city = df['市'][0]
            country = df['区'][0]
            area = df['地址'][0]
            if area:
                level1,level2 = split_p(area)
                print(prov, city, country, level1,level2)
        except:
            print(area)

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)
    # print(seg_list)

    return seg_list

def split_p(text):
    key1 = ['镇','乡','堡','街']
    key2 = ['村','路','区']

    level1 = None
    mark1 = 0
    level2 = None
    mark2 = 0
    for k in key1:
        mark1 = text.find(k)
        if mark1 !=-1:
            level1 = text[:mark1+1]
            break

    for k in key2:
        mark2 = text.find(k)
        if mark2 !=-1:
            level2 = text[mark1+1:mark2+1]
            break

    # print('level1',level1, 'level2',level2)

    return level1,level2


    # zhen = text.find('镇')
    # if  zhen != -1:
    #     print('镇',text[:zhen+1])
    #
    # xiang = text.find('乡')
    # if  xiang != -1:
    #     print('乡',text[:xiang+1])
    #
    # cun = text.find('村')
    # if cun != -1:
    #     print('村',text[zhen+1:cun+1])
    
if __name__ == '__main__':
    mysql_read()
    # sp = jieba_split('云南省昆明市石林彝族自治县长湖镇所各邑村委会所各邑村161号')
    # print(sp)
    #
    # location_str = ["云南省昆明市石林彝族自治县长湖镇所各邑村委会所各邑村161号"]
    # df = cpca.transform(location_str)
    # print(df)
    #
    # split_p('沐川县炭库乡石碑村8组19号')