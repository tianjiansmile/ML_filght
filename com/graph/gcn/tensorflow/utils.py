# -*- coding: utf-8 -*-
import scipy.sparse as sp
import networkx as nx
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    # The zeroth element of the tuple contains the cell location of each
    # non-zero value in the sparse matrix, each element looks like[i,j]
    # The first element of the tuple contains the value at each cell location
    # in the sparse matrix
    # The second element of the tuple contains the full shape of the sparse
    # matrix
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



def preprocess_adj(adj):
    '''
    Symmetrically normalize adjacency matrix.
    Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    '''
    adj_tilde = adj + np.identity(n=adj.shape[0])
    #np.squeeze()--从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = sparse_to_tuple(sp.coo_matrix(adj_norm))
    return adj_norm_tuple

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = sp.coo_matrix(features)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def build_label(Graph):
    '''
    Graph--Graph class defined in graph.py
    '''
    g = Graph
    G = g.G
    nodes = g.node_list
    look_up = g.look_up
    labels = []
    label_dict = {}
    label_id = 0
    for node in nodes:
        labels.append((node,G.nodes[node]['label']))
        for l in G.nodes[node]['label']:
            if l not in label_dict:
                label_dict[l] = label_id
                label_id += 1
    label_mat = np.zeros((len(labels),label_id))
    for node,l in labels:
        node_id = look_up[node]
        for ll in l:
            l_id = label_dict[ll]
            label_mat[node_id][l_id] = 1
    return label_mat,label_dict


def preprocess_labels(Graph,labels,clf_ratio):
    '''actually, preprocess label mask'''
    train_percent = clf_ratio
    g = Graph
    node_size = g.node_size
    look_up = g.look_up
    training_size = int(train_percent * node_size)

    state = np.random.get_state()
    np.random.seed(0)
    shuffle_indices = np.random.permutation(np.arange(node_size))
    np.random.set_state(state)

    def sample_mask(begin,end):
        mask = np.zeros(node_size)
        for i in range(begin, end):
            mask[shuffle_indices[i]] = 1
        return mask

    train_mask = sample_mask(0,training_size - 100)
    val_mask = sample_mask(training_size - 100, training_size)
    test_mask = sample_mask(training_size, node_size)
    return train_mask,val_mask,test_mask

def preprocess_data(Graph,clf_ration,has_features=True):
    '''
    Graph--Graph class defined in graph.py
    clf_ratio--the ratio of labels data for training; the default is 0.5;
    '''
    g = Graph # Graph class
    G = g.G # networkx graph object
    nodes = g.node_list
    # process node adj_matrix
    adj = nx.to_numpy_matrix(G)
    adj = preprocess_adj(adj)
    # process node features
    if has_features == True:
        features = np.vstack([G.nodes[i]['feature']
                for i in g.node_list])
        features = preprocess_features(features)
    else:
        features = g.features
    # preprocess node label masks
    labels,label_dict = build_label(g)
    train_mask,val_mask,test_mask = preprocess_labels(g,labels,clf_ration)
    return adj,labels,features,train_mask,val_mask,test_mask









