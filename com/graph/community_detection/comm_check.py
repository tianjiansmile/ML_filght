from com.graph.community_detection import pylouvain as lou
from com.graph.embedding.deepwalk.loan import neo4j_subgraph
from neo4j.v1 import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from com.jian.uml import kmeans as KM
import seaborn as sns

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

def test_loan_network(comm):

    # pyl = lou.PyLouvain.from_file('data/3229132edgelist.txt')
    pyl,node_dict = lou.PyLouvain.from_weight_file('data/'+comm+'edgelist.txt')
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
def get_features(nid):
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


#   对分团之后的团体进行历史特征评估
def loan_behivior_check(parts):
    feature_pool = {}
    community_pool = {}
    nodes = []
    count = 0
    # 查询特征
    for par in parts:
        community_pool[count] = []
        for p in par:
            sex,age,apply,approve,overdue,loanamount,max_overdue = get_features(p)
            community_pool[count].append((p,sex,age,apply,approve,overdue,loanamount,max_overdue))
        count+=1
    # print(community_pool)

    print('开始计算团特征')
    # 计算每一个团的特征
    node_count = [];gender_rate = [];max_age_diff = []
    avg_age = [];avg_apply = [];avg_approve = [];avg_overdue = []
    avg_loanamount = [];avg_maxoverdue = [];approve_rate = [];overdue_rate = []
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

            feature_pool[par] = feature
            nodes.append(par)

        # print(par,feature)

    # 将字典数据转换为pandas的数据用于分析
    pd_data = {'id':nodes,'node_count':node_count,'gender_rate':gender_rate,'max_age_diff':max_age_diff,
               'avg_age':avg_age,'avg_apply':avg_apply,'avg_approve':avg_approve,'avg_overdue':avg_overdue,
               'avg_loanamount':avg_loanamount,'avg_maxoverdue':avg_maxoverdue,'approve_rate':approve_rate,
               'overdue_rate':overdue_rate}

    col_list = ['node_count', 'gender_rate','max_age_diff',
                'avg_age','avg_apply','avg_approve','avg_overdue',
                'avg_loanamount','avg_maxoverdue','approve_rate','overdue_rate']


    df = pd.DataFrame(pd_data, columns=col_list)

    return df,col_list

# 对每一个特征进行简单分析
def data_check(train):
    print(train.shape)
    # print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    print(pd.isnull(train).values.any())

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
    plt.plot(train['id'], train['gender_rate'])
    plt.title('gender_rate values per id')
    plt.xlabel('id')
    plt.ylabel('gender_rate')
    plt.legend()
    plt.show()
    # # 基本上，偏度度量了实值随机变量的均值分布的不对称性。让我们计算损失的偏度：
    # stats.mstats.skew(train['loss']).data
    #
    # # 数据确实是倾斜的  对数据进行对数变换通常可以改善倾斜，可以使用 np.log
    # stats.mstats.skew(np.log(train['loss'])).data
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(16, 5)
    # ax1.hist(train['loss'], bins=50)
    # ax1.set_title('Train Loss target histogram')
    # ax1.grid(True)
    # ax2.hist(np.log(train['loss']), bins=50, color='g')
    # ax2.set_title('Train Log Loss target histogram')
    # ax2.grid(True)
    # plt.show()

    # 查看所有连续变量的分布
    # train[cont_features].hist(bins=50, figsize=(16, 12))

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
    label_count = 3

    # 2类
    km = KMeans(n_clusters=label_count).fit(node_pos)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer.sort_values('cluster')
    beer['tn1'] = node_pos[:,0]
    beer['tn2'] = node_pos[:,1]
    # beer['tn3'] = node_pos[:,2]

    beer.drop(w, axis=1, inplace=True)
    centers = beer.groupby("cluster").mean().reset_index()

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

def plot_2D(color_idx,new_df,centers,label_count):
    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)

    plt.scatter(centers.tn1, centers.tn2, linewidths=label_count, marker='+', s=300, c='black')

    plt.legend()
    plt.show()

# 对每一个特征画图了解分布情况
def plot_check(df ):
    df['node_count'].plot()
    df['avg_maxoverdue'].plot()
    plt.show()

    df['gender_rate'].plot()
    plt.show()

    df['max_age_diff'].plot()
    df['avg_age'].plot()
    df['node_count'].plot()
    # plt.show()
    df['avg_apply'].plot()
    plt.show()

    df['avg_loanamount'].plot()
    df['node_count'].plot()
    plt.show()

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
        if int(age) > max_age: max_age=int(age)
        if int(age) < min_age: min_age=int(age)
        avg_age+=int(age)
        avg_apply+=int(apply)
        avg_approve+=int(approve)
        avg_overdue+=int(overdue)
        avg_loanamount += float(loanamount)
        avg_maxoverdue += float(max_overdue)

    # print(comm)
    # print('count',count,'male',male,'famale',famale,'max_age',max_age,'min_age',min_age)
    # 男女比 默认 全为男性
    gender_rate = 0
    if count is not 0:
        gender_rate = round(male/count,2)
    # 最大年龄差
    max_age_diff = max_age - min_age

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
    if count is not 0:
        avg_age = round(avg_age/count,2)
        avg_apply = round(avg_apply / count, 2)
        avg_approve = round(avg_approve / count, 2)
        avg_overdue = round(avg_overdue / count, 2)
        avg_loanamount = round(avg_loanamount / count, 2)
        avg_maxoverdue = round(avg_maxoverdue / count, 2)

        return [node_count,gender_rate,max_age_diff,avg_age,avg_apply,avg_approve,
                avg_overdue,avg_loanamount,avg_maxoverdue,approve_rate,overdue_rate]


if __name__ == '__main__':
    my_neo4j = neo4j_subgraph.Neo4jHandler(driver)

    comm = '9375315'
    comm = '3229132'
    parts = test_loan_network(comm)
    # 团特征计算
    df,col_list = loan_behivior_check(parts)

    # 折线图查看每一维度分布
    # plot_check(df)

    # 降维
    # plot_embeddings(feature_pool,nodes)

    X = df[col_list]

    # 特征之间的相关性
    # feature_col(X)

    # 特征数据分析
    data_check(df)

    # 聚类评估：轮廓系数（Silhouette Coefficient ）
    # 通过轮廓系数 来确定最优的分类个数，越大越好
    # KM.silhouette_coefficient(X)

    # 无监督分类
    # kmeans(X,col_list,df)