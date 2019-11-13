# -*- coding: utf-8 -*-
import scipy.sparse
import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from com.graph.gcn.tensorflow import layers as lg
from com.graph.gcn.tensorflow import utils as us
from sklearn.manifold import TSNE

# 本脚本主要实现，GCN，通过对karate数据的图卷积操作，衍生出节点的特征
# 邻接矩阵：A， 特征数据: X  单位矩阵：I  度矩阵：D
# 从数学上看 A*X就可以聚合当前节点的一阶领域节点的特征和，
# (A+I)*X可以聚合自身节点以及邻居节点的特征，
# D**-1*A*X 可以将特征归一化，这一步参考拉普拉斯矩阵分解

def gcn():
    g = nx.read_edgelist('karate.edgelist', nodetype=int, create_using=nx.Graph())

    adj = nx.to_numpy_matrix(g)

    # Get important parameters of adjacency matrix
    n_nodes = adj.shape[0]

    # 得到具有自环的邻接矩阵 A_hat
    adj_tilde = adj + np.identity(n=n_nodes)
    # print(adj_tilde)

    # 构造度矩阵 D_hat 用于聚合每一个节点的邻居以及自己的特征
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))

    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
    # diag 获取对角线元素
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    # dot 矩阵乘法
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    # print(adj_norm)

    adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))

    # print(adj_norm_tuple)

    # Features are just the identity matrix 特征以单位矩阵表示
    feat_x = np.identity(n=n_nodes)
    feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))





    print(feat_x_tuple)

    ph = {
        'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
        'x': tf.sparse_placeholder(tf.float32, name="x")}

    l_sizes = [32, 16, 8]

    o_fc1 = lg.GraphConvLayer(input_dim=feat_x.shape[-1],
                              output_dim=l_sizes[0],
                              name='fc1',
                              act=tf.nn.tanh)(adj_norm=ph['adj_norm'],
                                              x=ph['x'], sparse=True)

    o_fc2 = lg.GraphConvLayer(input_dim=l_sizes[0],
                              output_dim=l_sizes[1],
                              name='fc2',
                              act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

    o_fc3 = lg.GraphConvLayer(input_dim=l_sizes[1],
                              output_dim=l_sizes[2],
                              name='fc3',
                              act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feed_dict = {ph['adj_norm']: adj_norm_tuple,
                 ph['x']: feat_x_tuple}

    outputs = sess.run(o_fc3, feed_dict=feed_dict)
    print(outputs.shape)
    nodes = list(g.nodes())
    labels = node2label(nodes)
    return outputs,labels,nodes

def node2label(nodes):
    d = {}
    res = []
    with open("karate.node", 'r') as f:
        lines = f.readlines()
    for line in lines:
        node, label = line.strip().split()
        d[int(node)] = int(label)
    for node in nodes:
        res.append(d[node])
    return res


def node_id(nodes):
    res = []
    for i in nodes:
        res.append(nodes.index(i))
    return res


def emb_reduction(embeddings):
    print("Embedding shape:", embeddings.shape)
    # TSNE's parameter perplexity maybe useful for visualization.
    tsne = TSNE(n_components=2, perplexity=10, init='pca', random_state=0, n_iter=5000, learning_rate=0.1)
    emb= tsne.fit_transform(embeddings)
#    print("After feature reduction:", emb_2.shape)
    return emb

def plot_embedding(X, nodes, labels):
    x= X[:, 0]
    y= X[:, 1]
    colors = []
    d = {0:'tomato', 1:'blue', 2:'lightgreen', 3:'lightgray'}
    for i in range(len(labels)):
        colors.append(d[labels[i]])
    plt.scatter(x, y, s=200, c=colors)
    for x,y, node in zip(x, y, nodes):
        plt.text(x, y, node, ha='center', va='center', fontsize=8)
    plt.show()

if __name__ == '__main__':
    # 预训练
    outputs, labels, nodes = gcn()
    nodes = node_id(nodes)
    # 降维
    emb = emb_reduction(outputs)
    print(emb)
    # 可视化
    plot_embedding(emb, nodes, labels)