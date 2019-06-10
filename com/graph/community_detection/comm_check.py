from com.graph.community_detection import pylouvain as lou

def test_loan_network():

    # pyl = lou.PyLouvain.from_file('data/loan_edgelist1.txt')
    pyl,node_dict = lou.PyLouvain.from_weight_file('data/loan_edgelist1.txt')
    # print(node_dict)
    partition, q = pyl.apply_method()
    # Q就是模块度，模块度越大则表明社区划分效果越好。Q值的范围在[-0.5,1），论文表示当Q值在0.3~0.7之间时，说明聚类的效果很好
    # print(partition, q)
    parts = []
    for p in partition:
        temp = []
        for a in p:
            if node_dict.get(a):
                temp.append(node_dict.get(a))
        parts.append(temp)

    print(parts, q)
    print(len(parts))
if __name__ == '__main__':
    test_loan_network()