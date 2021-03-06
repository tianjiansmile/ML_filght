
from com.graph.community_detection import pylouvain as lou
import unittest
import math

class PylouvainTest(unittest.TestCase):

    def test_karate_club(self):
        pyl = lou.PyLouvain.from_weight_file('data/3229132edgelist.txt')
        # print(pyl.nodes)
        partition, q = pyl.apply_method()
        # Q就是模块度，模块度越大则表明社区划分效果越好。Q值的范围在[-0.5,1），论文表示当Q值在0.3~0.7之间时，说明聚类的效果很好
        print(partition,q)

if __name__ == '__main__':
    pass
    # file = 'data/3229132edgelist.txt'
    # # file = 'data/little_edgelist.txt'
    # test_karate_club(file)