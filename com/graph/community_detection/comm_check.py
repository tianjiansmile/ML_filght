from com.graph.community_detection import pylouvain as lou
from com.graph.embedding.deepwalk.loan import neo4j_subgraph
from neo4j.v1 import GraphDatabase
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D as p3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from com.jian.uml import kmeans as KM
from scipy import stats
import seaborn as sns
import numpy
import requests

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

# 本脚本通过PyLouvain 算法对大社区进行团划分，然后计算团的特征
# 以团为单位分析不同团的表现情况，一种是利用降维将团无监督分类

# TSNE 降维
def TSNE_handle(embeddings,col):
    # 读取node id 和对应标签

    model = TSNE(n_components=col)
    node_pos = model.fit_transform(embeddings)

    return node_pos

def test_loan_network(comm,weight_mark):

    # pyl,node_dict = lou.PyLouvain.from_weight_file('data/md5_rels.csv')
    pyl,node_dict = lou.PyLouvain.from_weight_file('data/'+comm+'edgelist.txt',weight=weight_mark)
    # print(node_dict)
    partition, q = pyl.apply_method()
    # Q就是模块度，模块度越大则表明社区划分效果越好。Q值的范围在[-0.5,1），论文表示当Q值在0.3~0.7之间时，说明聚类的效果很好

    parts = []
    for p in partition:
        temp = []
        for a in p:
            if node_dict.get(a):
                temp.append(node_dict.get(a))
        parts.append(temp)

    print('模块度：', q,'长度：',len(parts))

    return parts


# 从网络中查询每一个节点的基础特征
def get_features(nid,my_neo4j):
    cypher = "match (p:person) where p.nid = '"+nid+"' return p.sex as sex," \
             "p.age as age,p.apply as apply,p.approve as approve," \
             "p.overdue as overdue,p.loanamount as loanamount,p.max_overdue as max_overdue"
    result = my_neo4j.cypherexecuter(cypher)
    sex = '-1'
    age = '-1'
    apply = '-1'
    approve = '-1'
    overdue = '-1'
    loanamount = '-1'
    max_overdue = '-1'
    for item in result:
        sex = item[0]
        age = item[1]
        apply = item[2]
        approve = item[3]
        overdue = item[4]
        loanamount = item[5]
        max_overdue = item[6]
    # print(sex,age,apply,approve,overdue,loanamount,max_overdue)
    return sex,age,apply,approve,overdue,loanamount,max_overdue

# 实时计算特征
def get_loan_feature(nid):
    all_dic = {}
    url = 'http://47.101.206.54:8023/getModelEncyLoanFeatures?identityNo=%s'
    res = requests.get(url % (nid))


    if res.status_code==200:
        all_list=[]
        res=res.json()
        result=res.get('result')
        features=result.get('features')
        if features:
            user_feature = features.get('user_feature')
            loan_behavior_feature = features.get('loan_behavior_feature')
            loan_merchant_feature = features.get('loan_merchant_feature')
            # print(loan_merchant_feature)


        return loan_behavior_feature,loan_merchant_feature
    else:
        return None

#   对分团之后的团体进行历史特征评估
def loan_behivior_check(parts,my_neo4j):
    feature_pool = {}
    community_pool = {}
    nodes = []
    count = 0
    # 查询特征
    for par in parts:
        community_pool[count] = []
        for p in par:
            sex,age,apply,approve,overdue,loanamount,max_overdue = get_features(p,my_neo4j)
            community_pool[count].append((p,sex,age,apply,approve,overdue,loanamount,max_overdue))
        count+=1
    # print(community_pool)

    print('开始计算团特征')
    # 计算每一个团的特征
    node_count = [];gender_rate = [];max_age_diff = []
    avg_age = [];avg_apply = [];avg_approve = [];avg_overdue = []
    avg_loanamount = [];avg_maxoverdue = [];approve_rate = [];overdue_rate = []
    avg_M3 = [];avg_pd10 = []
    for par in community_pool:
        comm = community_pool[par]
        # node_count,gender_rate,max_age_diff,avg_age,avg_apply,avg_approve,
        # avg_overdue,avg_loanamount,avg_maxoverdue,approve_rate,overdue_rate
        feature = feature_handler(comm)
        if feature: # 如果feature是空的说明这个团里的特征都是-1，没有更新到网络里
            node_count.append(feature[0])
            gender_rate.append(feature[1])
            max_age_diff.append(feature[2])
            avg_age.append(feature[3])
            avg_apply.append(feature[4])
            avg_approve.append(feature[5])
            avg_overdue.append(feature[6])
            avg_loanamount.append(feature[7])
            avg_maxoverdue.append(feature[8])
            approve_rate.append(feature[9])
            overdue_rate.append(feature[10])
            avg_M3.append(feature[11])
            avg_pd10.append(feature[12])

            feature_pool[par] = feature
            nodes.append(par)

        # print(par,feature)

    # 将字典数据转换为pandas的数据用于分析
    pd_data = {'id':nodes,'node_count':node_count,'gender_rate':gender_rate,'max_age_diff':max_age_diff,
               'avg_age':avg_age,'avg_apply':avg_apply,'avg_approve':avg_approve,'avg_overdue':avg_overdue,
               'avg_loanamount':avg_loanamount,'avg_maxoverdue':avg_maxoverdue,'approve_rate':approve_rate,
               'overdue_rate':overdue_rate,'avg_M3':avg_M3,'avg_pd10':avg_pd10}

    col_list = ['id','node_count', 'gender_rate','max_age_diff',
                'avg_age','avg_apply','avg_approve','avg_overdue',
                'avg_loanamount','avg_maxoverdue','approve_rate','overdue_rate','avg_M3','avg_pd10']


    df = pd.DataFrame(pd_data, columns=col_list)

    return df,col_list

# 实时查询用户的多头情况
def loan_behivior_check_pro(parts):
    # 查询特征
    feature_pool = []
    for p in parts:
        loan_behavior_feature,loan_merchant_feature = get_loan_feature(p)
        feature_pool.append((
            loan_merchant_feature.get('apply_mert_pdl_diff_1'),
            loan_merchant_feature.get('apply_mert_pdl_diff_2'),
            loan_merchant_feature.get('apply_mert_pdl_diff_3'),
            loan_merchant_feature.get('apply_mert_pdl_diff_4'),
            loan_merchant_feature.get('apply_mert_pdl_diff_5'),
            loan_merchant_feature.get('apply_mert_pdl_diff_6'),
            loan_merchant_feature.get('apply_mert_pdl_diff_7'),
            loan_merchant_feature.get('apply_mert_pdl_diff_8'),
            loan_merchant_feature.get('apply_mert_pdl_diff_9'),
            loan_merchant_feature.get('apply_mert_pdl_diff_10'),
            loan_merchant_feature.get('apply_mert_pdl_diff_11'),
            loan_merchant_feature.get('apply_mert_pdl_sum'),

            loan_merchant_feature.get('apply_mert_int_diff_1'),
            loan_merchant_feature.get('apply_mert_int_diff_2'),
            loan_merchant_feature.get('apply_mert_int_diff_3'),
            loan_merchant_feature.get('apply_mert_int_diff_4'),
            loan_merchant_feature.get('apply_mert_int_diff_5'),
            loan_merchant_feature.get('apply_mert_int_diff_6'),
            loan_merchant_feature.get('apply_mert_int_diff_7'),
            loan_merchant_feature.get('apply_mert_int_diff_8'),
            loan_merchant_feature.get('apply_mert_int_diff_9'),
            loan_merchant_feature.get('apply_mert_int_diff_10'),
            loan_merchant_feature.get('apply_mert_int_diff_11'),
            loan_merchant_feature.get('apply_mert_int_sum'),


        ))

    return feature_pool



# 对每一个特征进行简单分析
def data_check(train):

    print(train.shape)
    # 均值mean 方差std 最大最小 min max
    # print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    # print(pd.isnull(train).values.any())

    # 数据的整体状况
    # print(train.info())

    # 查看离散变量的个数
    cat_features = list(train.select_dtypes(include=['object']).columns)
    print("Categorical: {} features".format(len(cat_features)))
    # 查看连续变量的个数
    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont  is not  'id']
    print("Continuous: {} features".format(len(cont_features)))

    plt.figure(figsize=(16, 8))
    # # 我们看一下损失值的分布 损失值中有几个显著的峰值表示严重事故。这样的数据分布，使得这个功能非常扭曲导致的回归表现不佳。
    plt.plot(train['node_count'], train['gender_rate'])
    plt.title('gender_rate values per id')
    plt.xlabel('node_count')
    plt.ylabel('gender_rate')
    plt.legend()
    plt.show()
    # # 基本上，偏度度量了实值随机变量的均值分布的不对称性。让我们计算损失的偏度：
    bis = stats.mstats.skew(train['gender_rate']).data
    print(bis)
    #
    # # 数据确实是倾斜的  对数据进行对数变换通常可以改善倾斜，可以使用 np.log
    # stats.mstats.skew(np.log(train['gender_rate'])).data
    # #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(16, 5)
    # ax1.hist(train['gender_rate'], bins=50)
    # ax1.set_title('Train gender_rate target histogram')
    # ax1.grid(True)
    # ax2.hist(np.log(train['gender_rate']), bins=50, color='g')
    # ax2.set_title('Train Log gender_rate target histogram')
    # ax2.grid(True)
    #
    # 查看所有连续变量的分布,柱状图
    train[cont_features].hist(bins=50, figsize=(16, 12))
    plt.show()

 # 特征之间的相关性
def feature_col(X):
    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = X.corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

# 无监督分类
def kmeans(X,w,beer):

    # 映射之后的维度
    col = 2
    node_pos = TSNE_handle(X,col)

    # 分类个数
    label_count = 7

    # 2类
    km = KMeans(n_clusters=label_count).fit(node_pos)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer.sort_values('cluster')
    beer['tn1'] = node_pos[:,0]
    beer['tn2'] = node_pos[:,1]
    # beer['tn3'] = node_pos[:,2]

    # w.remove('id')
    # w.remove('node_count')
    # beer.drop(w, axis=1, inplace=True)
    centers = beer.groupby("cluster").mean().reset_index()

    # 获得pd所有列名
    # print(list(beer.columns))
    # print(beer[['id','node_count','cluster']])
    cluster = beer.groupby("cluster")
    # print(cluster.describe())
    # 查看每一个分类的特征均值分布，验证
    print((cluster.aggregate({"avg_apply": numpy.mean,"avg_approve": numpy.mean,
                              "avg_overdue": numpy.mean,"avg_maxoverdue": numpy.mean,"avg_M3": numpy.mean})))
    print((cluster.aggregate({"approve_rate": numpy.mean, "max_age_diff": numpy.mean, "gender_rate": numpy.mean,
                              "node_count": numpy.mean})))

    color_idx = {}
    for i in range(label_count):
        color_idx.setdefault(i, [])

    count = 0
    for lab in beer['cluster']:
        for i in range(label_count):
            if lab == i:
                color_idx[i].append(count)
        count += 1

    # print(color_idx)

    plot_2D(color_idx, node_pos, centers, label_count)
    # plot_3D(color_idx, node_pos, centers, label_count)

def plot_2D(color_idx,new_df,centers,label_count):
    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)

    plt.scatter(centers.tn1, centers.tn2, linewidths=label_count, marker='+', s=300, c='black')

    plt.legend()
    plt.show()

def plot_3D(color_idx,new_df,centers,label_count):

    p3d = plt.figure().add_subplot(111, projection='3d')
    for c, idx in color_idx.items():
        p3d.scatter(new_df[idx, 0], new_df[idx, 1], new_df[idx, 2], zdir='z', label=c,s=30, c=None, depthshade=True)
    plt.legend()
    plt.show()
# 对每一个特征画图了解分布情况
def plot_check(df ):
    df['node_count'].plot()
    df['avg_maxoverdue'].plot()
    plt.show()
    #
    df['gender_rate'].plot()
    plt.show()

    df['max_age_diff'].plot()
    df['avg_age'].plot()
    df['node_count'].plot()
    plt.show()
    df['avg_apply'].plot()
    plt.show()

    # df['avg_loanamount'].plot()
    # df['node_count'].plot()
    # plt.show()

# 将embedding向量转化为二维空间数据
def plot_embeddings(embeddings, nodes):
    # 读取node id 和对应标签
    X = nodes

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    # 降维
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault('0', [])
        color_idx['0'].append(i)

    # 利用list的有序性，将每一个节点的降维特征映射到画布上
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

# 对特征进行计算
def feature_handler(comm):
    count = 0
    male = 0
    famale = 0
    max_age = 0
    min_age = 100
    avg_age = 0
    avg_apply =0
    avg_approve = 0
    avg_overdue = 0
    avg_loanamount = 0.0
    avg_maxoverdue = 0
    avg_M3 = 0
    avg_pd10 = 0

    for item in comm:
        sex = item[1]
        age = item[2]
        apply = item[3]
        approve = item[4]
        overdue = item[5]
        loanamount = item[6]
        max_overdue = item[7]
        if sex=='-1' and age=='-1':
            continue

        count += 1
        if sex=='1': male+=1
        if sex=='2': famale+=1
        if int(age) > 0:
            if int(age) > max_age: max_age=int(age)
            if int(age) < min_age: min_age=int(age)
        avg_age+=int(age)
        avg_apply+=int(apply)
        avg_approve+=int(approve)
        avg_overdue+=int(overdue)
        avg_loanamount += float(loanamount)
        avg_maxoverdue += float(max_overdue)

        if float(max_overdue)>=90:
            avg_M3+=1

        if float(max_overdue)>=10:
            avg_pd10+=1

    # print(comm)
    # print('count',count,'male',male,'famale',famale,'max_age',max_age,'min_age',min_age)
    # 男女比 默认 全为男性
    # gender_rate = 0
    # max_age_diff = 0
    if count is not 0:
        gender_rate = round(male/count,2)

        # 最大年龄差
        max_age_diff = max_age - min_age

        # print('count',count,'male',male,'famale',famale,'max_age',max_age,'min_age',min_age,'max_age_diff',max_age_diff)

        # 通过率
        approve_rate = 0
        # 逾期率
        overdue_rate = 0
        if avg_apply is not 0:
            approve_rate = round(avg_approve / avg_apply, 2)
        if avg_approve is not 0:
            overdue_rate = round(overdue_rate / avg_approve, 2)

        #  节点数
        node_count = len(comm)
        # print('avg_age', avg_age, 'avg_apply', avg_apply, 'avg_approve', avg_approve,
        #       'avg_overdue', avg_overdue, 'avg_loanamount', avg_loanamount, 'avg_maxoverdue', avg_maxoverdue)
        avg_age = round(avg_age/count,2)
        avg_apply = round(avg_apply / count, 2)
        avg_approve = round(avg_approve / count, 2)
        avg_overdue = round(avg_overdue / count, 2)
        avg_loanamount = round(avg_loanamount / count, 2)
        avg_maxoverdue = round(avg_maxoverdue / count, 2)
        avg_M3 = round(avg_M3 / count,2)
        avg_pd10 = round(avg_pd10 / count, 2)

        return [node_count,gender_rate,max_age_diff,avg_age,avg_apply,avg_approve,
                avg_overdue,avg_loanamount,avg_maxoverdue,approve_rate,overdue_rate,avg_M3,avg_pd10]

# 定为一些有欺诈风险的团体，
# avg_approve: 高 overdue：高  通过率高的一些团体，逾期的指标也高的话，这个团体欺诈风险很高
# avg_approve: 低 overdue：高  通过率一般，但是逾期指标较高，团体潜在欺诈风险较高
# 通过指标：avg_approve均值位0.36  approve_rate均值位0.11
# 逾期指标：avg_overdue 均值位0.25 avg_maxoverdue均值位9.72 avg_M3均值位0.04  avg_pd10 均值位0.06
def locate_risk_community(df,part):
    # 团平均通过次数 baseline
    avg_approve = 0.36
    # 团申请通过比
    approve_rate = 0.11

    # 最大逾期天数均值
    avg_maxoverdue = 9.72
    # 团平均逾期次数
    avg_overdue = 0.25
    # 团平均M3次数
    avg_M3 = 0.04

    # 团平均M3次数
    avg_M3 = 0.04
    avg_pd10 = 0.06

    # 对数据进行过滤，找到通过率和最大逾期天数都高于平均水平的数据
    new1 = df.query('avg_maxoverdue > 9.72').query('avg_approve > 0.36') \
                    .sort_values('avg_maxoverdue') \
                    .tail(20)

    new2 = df.query('avg_maxoverdue < 9.72').query('avg_approve < 0.36') \
        .sort_values('avg_maxoverdue')[:40] \

    #
    # data_check(new1)

    # plot_check(new1)

    print(new1[['avg_apply','avg_approve','avg_overdue',
                'avg_M3', 'avg_maxoverdue','node_count']])

    print(new2[['avg_apply', 'avg_approve', 'avg_overdue',
                'avg_M3', 'avg_maxoverdue', 'node_count']])

    # print(part[199 ])
    # # # 打印高风险团体成员
    # print(part[202])
    # print(part[99])

def write_parts(parts):
    with open('all_part.txt','w') as wf:

        count = 0
        for p in parts:
            print(p)
            wf.write(str(count)+'\t'+str(p)+'\n')
            count+=1

if __name__ == '__main__':
    import time
    starttime = time.time()
    my_neo4j = neo4j_subgraph.Neo4jHandler(driver)

    comm = '3229132'
    # comm = '9375315'
    # comm = '3920885'
    # comm = '4818846'
    # # comm = '7875775'
    # comm = '8998331'

    # 获得团体编号，以及团体成员
    # weight_mark 1 默认使用权重
    parts = test_loan_network(comm,1)

    # write_parts(parts)

    # 团特征计算
    df,col_list = loan_behivior_check(parts,my_neo4j)

    # 团实时特征计算
    # df, col_list = loan_behivior_check_pro(parts)
    #
    # # 折线图查看每一维度分布
    # # plot_check(df)
    #
    # # 降维
    # # plot_embeddings(feature_pool,nodes)
    #
    X = df[col_list]
    #
    # # 特征之间的相关性
    # # feature_col(X)
    #
    # for col in df.columns:
    #     print(col,"该列数据的均值位%.2f" % df[col].mean())  # 计算每列均值
    # #
    # # print('方差：',df.std(ddof=0))
    # # 特征数据分析
    # # data_check(df)
    #
    # # 定位一些有欺诈风险的团体
    locate_risk_community(df,parts)
    #
    # # 聚类评估：轮廓系数（Silhouette Coefficient ）
    # # 通过轮廓系数 来确定最优的分类个数，越大越好
    # #
    # # KM.silhouette_coefficient(X)
    #
    # # 无监督分类
    # X.drop(['id'], axis=1, inplace=True)
    # kmeans(X,col_list,df)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)