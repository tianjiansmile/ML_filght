#  coding: utf-8
import xlsxwriter
import json
import pandas as pd
import requests

# 对地址数据调用接口，获得地址特征

addr_all_dict = []

def map_addr_feature(addr):
    if addr:
        try:
            res = requests.get('http://127.0.0.1:5000/getAddrFeatures?addr=%s' % addr)
            feature_pool = {}
            if res.status_code==200:
                all_list=[]
                res=res.json()
                result=res.get('result')
                feature_pool.update(result)
            # print(feature_pool.get('approve_rate_prov'),feature_pool.get('overdue_rate_prov'))
            addr_all_dict.append(feature_pool)
        except:
            print('error')

        return feature_pool.get('addr_approve_rate_prov')
    else:
        return None


if __name__ == '__main__':
    merchant_id = '3410'
    file = 'data/'+merchant_id+'addr_label.csv'
    data = pd.read_csv(file,encoding='utf-8')

    data['addr_approve_rate_prov'] = data['addr'].map(map_addr_feature)

    data.to_excel('data/'+merchant_id+'addr_lda_feature.xlsx')
    #
    print("准备写表格")
    f = xlsxwriter.Workbook('data/temp.xlsx')
    sheet1 = f.add_worksheet(u'sheet1')
    file_dict = {'addr_approve_rate_prov': 0.0, 'addr_approve_rate_city': 0.0, 'addr_approve_rate_country':0.0,'addr_approve_rate_area':0.0,
                 'addr_overdue_rate_prov':0.0,'addr_overdue_rate_city':0.0,'addr_overdue_rate_country':0.0,'addr_overdue_rate_area':0.0,
                 'addr_avg_apply_prov': 0.0, 'addr_avg_apply_city': 0.0, 'addr_avg_apply_country': 0.0,'addr_avg_apply_area':0.0,
                 'addr_avg_approve_prov': 0.0, 'addr_avg_approve_city': 0.0, 'addr_avg_approve_country': 0.0,'addr_avg_approve_area':0.0,
                 'addr_avg_overdue_prov': 0.0, 'addr_avg_overdue_city': 0.0, 'addr_avg_overdue_country': 0.0,'addr_avg_overdue_area':0.0,
                 'addr_avg_loanamount_prov': 0.0, 'addr_avg_loanamount_city': 0.0, 'addr_avg_loanamount_country': 0.0,'addr_avg_loanamount_area':0.0,

                 'addr_avg_pd3_prov': 0.0, 'addr_avg_pd3_city': 0.0, 'addr_avg_pd3_country': 0.0,'addr_avg_pd3_area':0.0,
                 'addr_avg_pd7_prov': 0.0, 'addr_avg_pd7_city': 0.0, 'addr_avg_pd7_country': 0.0,'addr_avg_pd7_area':0.0,
                 'addr_avg_pd10_prov': 0.0, 'addr_avg_pd10_city': 0.0, 'addr_avg_pd10_country': 0.0,'addr_avg_pd10_area':0.0,
                 'addr_avg_pd14_prov': 0.0, 'addr_avg_pd14_city': 0.0, 'addr_avg_pd14_country': 0.0,'addr_avg_pd14_area':0.0,
                 'addr_avg_M1_prov': 0.0, 'addr_avg_M1_city': 0.0, 'addr_avg_M1_country': 0.0,'addr_avg_M1_area':0.0,
                 'addr_avg_M2_prov': 0.0, 'addr_avg_M2_city': 0.0, 'addr_avg_M2_country': 0.0,'addr_avg_M2_area':0.0,
                 'addr_avg_M3_prov': 0.0, 'addr_avg_M3_city': 0.0, 'addr_avg_M3_country': 0.0,'addr_avg_M3_area':0.0,

                 'addr_pd3_rate_prov': 0.0, 'addr_pd3_rate_city': 0.0, 'addr_pd3_rate_country': 0.0,'addr_pd3_rate_area':0.0,
                 'addr_pd7_rate_prov': 0.0, 'addr_pd7_rate_city': 0.0, 'addr_pd7_rate_country': 0.0,'addr_pd7_rate_area':0.0,
                 'addr_pd10_rate_prov': 0.0, 'addr_pd10_rate_city': 0.0, 'addr_pd10_rate_country': 0.0,'addr_pd10_rate_area':0.0,
                 'addr_pd14_rate_prov': 0.0, 'addr_pd14_rate_city': 0.0, 'addr_pd14_rate_country': 0.0,'addr_pd14_rate_area':0.0,
                 'addr_M1_rate_prov': 0.0, 'addr_M1_rate_city': 0.0, 'addr_M1_rate_country': 0.0,'addr_M1_rate_area':0.0,
                 'addr_M2_rate_prov': 0.0, 'addr_M2_rate_city': 0.0, 'addr_M2_rate_country': 0.0,'addr_M2_rate_area':0.0,
                 'addr_M3_rate_prov': 0.0, 'addr_M3_rate_city': 0.0, 'addr_M3_rate_country': 0.0,'addr_M3_rate_area':0.0
                 }

    s = 0
    for key in file_dict:
        sheet1.write(0, s, key)
        s += 1
    print("表头设置完成")

    i = 1
    for da in addr_all_dict:
        c = 0
        for k in file_dict.keys():
            sheet1.write(i, c, da[k])
            # print(i, c, da[k])
            c+=1
        i = i + 1
        print("正在写入第%s行" % i)
    print("内容写入完成")
    f.close()  # 保存文件

