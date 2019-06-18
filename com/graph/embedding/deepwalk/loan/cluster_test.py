import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from mpl_toolkits.mplot3d import Axes3D as p3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# 对社交网络中的一个社区的embedding向量进行无监督分类

def plot_2D(color_idx,new_df,centers,label_count):
    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)

    plt.scatter(centers.tn1, centers.tn2, linewidths=label_count, marker='+', s=300, c='black')
    # plt.scatter(centers.tn1, centers.tn2, marker='+', s=300, c='black')

    plt.legend()
    plt.show()

def plot_3D(color_idx,new_df,centers,label_count):

    p3d = plt.figure().add_subplot(111, projection='3d')
    for c, idx in color_idx.items():
        p3d.scatter(new_df[idx, 0], new_df[idx, 1], new_df[idx, 2], zdir='z', label=c,s=30, c=None, depthshade=True)
    plt.legend()
    plt.show()

def kmeans(X,w,label_count):

    # 映射之后的维度
    col = 2
    node_pos = TSNE_handle(X,col)

    # label_count = 100

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

    # plot_2D(color_idx, node_pos, centers, label_count)

    # plot_3D(color_idx, node_pos, centers, label_count)

    return beer

def dbscan(X):

    new_df = TSNE_handle(X)

    db = DBSCAN(eps=10, min_samples=2).fit(new_df)

    # 分类结果
    labels = db.labels_
    print(labels)
    beer['cluster'] = labels

    color_idx = {0: [], 1: []}
    color_idx.setdefault(0, [])
    count = 0
    for lab in beer['cluster']:

        if lab == 0:
            color_idx[0].append(count)
        else:
            color_idx[1].append(count)
        count += 1

    print(color_idx)

    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)
    plt.legend()
    plt.show()

# PCA 降维
def pca_handle(new_df):
    pca = PCA(n_components=2)
    new_pca = pd.DataFrame(pca.fit_transform(new_df))

    return new_pca

# TSNE 降维
def TSNE_handle(embeddings,col):
    # 读取node id 和对应标签

    model = TSNE(n_components=col)
    node_pos = model.fit_transform(embeddings)

    return node_pos

# 对子图deepwalk训练得到的word2vec特征进行UML
def word_vec_test(beer,w,label_count):
    feature = ['v' + str(i) for i in range(1, 301)]
    # print(feature)
    X = beer[feature]
    kmeans(X,feature,label_count)
    # dbscan(X)


def label_to_file(beer):
    comm_list = []
    group = beer.groupby("cluster")
    for i in group.groups:
        nodes = group.groups[i]
        nodes = list(nodes)
        comm_list.append(nodes)

    return comm_list


if __name__ == '__main__':
    from com.graph.community_detection import comm_check
    from com.graph.embedding.deepwalk.loan import neo4j_subgraph
    from neo4j.v1 import GraphDatabase
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))
    my_neo4j = neo4j_subgraph.Neo4jHandler(driver)

    comm = '8998331'
    beer = pd.read_csv(comm+'word_vec.txt', sep=' ')

    w,v = beer.shape
    print(w,v)

    # 用无监督进行社区划分
    label_count = 371
    word_vec_test(beer,w,label_count)

    comm_list = label_to_file(beer)

    # 团特征计算
    df, col_list = comm_check.loan_behivior_check(comm_list,my_neo4j)

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
    comm_check.locate_risk_community(df, comm_list)
    #
    # # 聚类评估：轮廓系数（Silhouette Coefficient ）
    # # 通过轮廓系数 来确定最优的分类个数，越大越好
    # #
    # # KM.silhouette_coefficient(X)
    #
    # # 无监督分类
    X.drop(['id'], axis=1, inplace=True)
    comm_check.kmeans(X, col_list, df)