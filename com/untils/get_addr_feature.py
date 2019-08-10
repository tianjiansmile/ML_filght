import multiprocessing
import requests
import json
import pandas as pd
import numpy as np
import pymysql

#################################
# 最新接口

jinpan_url = "http://172.19.133.11:8023/getAddrFeatures?addr=%s"


df_all = pd.read_csv("D:/特征提取/嘉卡/live_addr_df.csv")
# df_all = df_all.head(8000)



print(df_all.shape)
print("唯一order_id",len(np.unique(df_all.order_id)))
# df_all.head(3)

df_all = df_all.drop_duplicates(subset=None, keep='first', inplace=False)

print(df_all.shape)

print(df_all.head())



########
# tupan-test 02
# 47.101.206.54(弹性)
# 172.19.133.11(私有)

write_folder = 'D:/特征提取/嘉卡/data/'
#
def save_feature(range_index):

    f = open(write_folder+"jiaka_addr_feature_%s.csv" %(str(range_index)),'w')
    # range_index = range(130,140)
    line_cnt = 0

    # 2542 231750373882732544 feature_size not eqal 1035

    for index in range_index:
        # index=2542
        # index=134 # 全空
        order_id = df_all.loc[index, "order_id"]
        live_addr = df_all.loc[index, "live_addr"]
        card_addr = df_all.loc[index, "card_addr"]


        try:


            feat_title = ["order_id"]
            feat_value = [str(order_id)]

            # feat_title.append('live_addr')
            # feat_value.append(live_addr)
            # feat_title.append('card_addr')
            # feat_value.append(card_addr)


            jinpan_request_url = jinpan_url % (live_addr)
            jinpan_response = requests.get(jinpan_request_url)
            live_features = json.loads(str(jinpan_response.content, 'utf-8')).get('result')
            # print(features.keys())

            for key,values in live_features.items():
                   feat_title.append('live_'+key)
                   feat_value.append(values)

            if card_addr:
                jinpan_request_url = jinpan_url % (card_addr)
            else:
                jinpan_request_url = jinpan_url % ('')
            jinpan_response = requests.get(jinpan_request_url)
            card_features = json.loads(str(jinpan_response.content, 'utf-8')).get('result')
            # print(features.keys())

            for key, values in card_features.items():
                feat_title.append('card_'+key)
                feat_value.append(values)

            # print(len(feat_title), len(feat_value), "\n")

            if len(feat_value) == 161:
                print(index, order_id)
                line_cnt = line_cnt +1
                if line_cnt == 1:
                   f.write(','.join(feat_title)+'\n')
                   f.write(','.join(list(map(lambda x:str(x),feat_value)))+'\n')
                else:
                   f.write(','.join(list(map(lambda x:str(x),feat_value)))+'\n')
            else:
                print(index, order_id, "feature_size not eqal 1035",len(feat_value))

        except Exception as e:
            raise  Exception
            print(index, order_id,e)

    f.close()




# save_feature(range(525130,525140))
# len_df = len(df_all)
len_df = len(df_all)
minmax_inds = [k for k in range(0, len_df, 20000)] + [len_df]
ls_inds = []
for k in range(len(minmax_inds) - 1):
    ls_inds.append(range(minmax_inds[k], minmax_inds[k + 1]))

print(ls_inds)


# save_feature(range(0, 208163))


pool = multiprocessing.Pool(10)
for iiinds in ls_inds:
    pool.apply_async(save_feature, [iiinds,])
# #
pool.close()
pool.join()



###############################

# print("\n合并数据\n")
#
# import os
#
# jp_feature_file_list = os.listdir(write_folder)
#
# jp_feature = pd.DataFrame()
# for jp_feature_file in jp_feature_file_list:
#     jp_feature_i = pd.read_csv(write_folder+jp_feature_file)
#     print(jp_feature_i.shape)
#     jp_feature = pd.concat([jp_feature,jp_feature_i],axis=0)
#
# print(jp_feature.shape)
# jp_feature.to_csv(write_folder + "a19_jinpan_feature.csv",index=False)





