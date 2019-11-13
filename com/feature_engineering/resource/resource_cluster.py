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
from sklearn.metrics import roc_auc_score
from com.risk_score import scorecard_functions_V3 as sf

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
    for lab in beer['overdueday']:
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



def label_to_file(beer):
    comm_list = []
    group = beer.groupby("cluster")
    for i in group.groups:
        nodes = group.groups[i]
        nodes = list(nodes)
        comm_list.append(nodes)

    return comm_list

# 将用户审核通过次数转换为是否审核通过
def map_label(x):
    if x > 3:
        return 1
    else:
        return 0

if __name__ == '__main__':
    # beer = pd.read_csv('data/3410_value.txt', sep=',')

    beer = pd.read_csv('data/resource_fea_label.csv', sep=',')
    beer.dropna(axis=0, how='any', inplace=True)

    print(pd.isnull(beer).values.any())

    # 将逾期次数转化为0，1标签
    beer['overdueday'] = beer['overdueday'].map(map_label)

    beer.drop(['order_id'], axis=1, inplace=True)

    cont_features = [cont for cont in list(beer.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['overdueday']]

    w, v = beer.shape
    print(w, v)
    print(cont_features)


    # 用无监督分类
    label_count = 2
    X = beer[cont_features]
    kmeans(X,cont_features,label_count)

    auc = roc_auc_score(beer['overdueday'], beer['cluster'])
    ks = sf.KS(beer, 'cluster', 'overdueday')
    print('准确度Area Under Curve auc', auc, '区分度 KS', ks)
    #
    # comm_list = label_to_file(beer)

    # # 折线图查看每一维度分布
    # plot_check(df)
    #
    # # 降维
    # # plot_embeddings(feature_pool,nodes)
    #
    # X = df[col_list]
    #
    # # 特征之间的相关性
    # # feature_col(X)
    # # 聚类评估：轮廓系数（Silhouette Coefficient ）
    # # 通过轮廓系数 来确定最优的分类个数，越大越好
    # #
    # # KM.silhouette_coefficient(X)
    #
    # # 无监督分类
    # X.drop(['id'], axis=1, inplace=True)
    # comm_check.kmeans(X, col_list, df)