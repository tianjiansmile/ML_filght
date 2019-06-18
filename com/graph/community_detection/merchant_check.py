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
from com.graph.community_detection import comm_check

# 主要用于检测团体用户的借贷行为是否存在同步性的情况，主要包括申请时间，机构迁移时间
if __name__ == '__main__':
    import time

    starttime = time.time()
    comm = '3229132'
    # comm = '9375315'
    # comm = '3920885'
    # comm = '4818846'
    # # comm = '7875775'
    # comm = '8998331'

    # 获得团体编号，以及团体成员
    # parts = comm_check.test_loan_network(comm)
    #
    # # 团特征计算
    # df, col_list = comm_check.loan_behivior_check(parts)
    #
    # # # 定位一些有欺诈风险的团体
    # comm_check.locate_risk_community(df, parts)

    temp = ['2c1911174961017fbd8de92724084ad0', '43607d50ca7bb4ccd0abebf7344fb6ec', '4b2910e5f1843f45cac78c14982e37ec', '567114a0fecb90c98f00aa9ef882118c', 'a80339987902913f7989049a6b298e40', 'c6dce326b5ea8fcd351b62bf31dea285', '859dccac971dc044846e310e6b5075d1', 'd369e7afcd7eaa9daaab154731d93830']
    feature_pool = comm_check.loan_behivior_check_pro(temp)
    # print(feature_pool)
    for i in feature_pool:
        print(i)

