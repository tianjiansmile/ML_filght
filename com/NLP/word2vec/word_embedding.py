import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as p3d
import pandas as pd
import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 本脚本旨在，预训练地址文本的word embedding 向量，
# 与LDA不同，word2vec可以学习到文本的上下文信息，这也是word embedding 特征与LDA主题特征的差异性

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(''.join(re.findall('[\u4e00-\u9fa5]+', text)))
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)

    return seg_list

# 分词成句
def write_sentence(live_addr,merchant_id):
    count = 1
    with open('data/'+merchant_id+'_sentence.txt', 'w',encoding="utf-8") as f:
        for docu in live_addr:
            try:
                if docu:
                    count += 1

                    # 分词
                    sentence = jieba_split(docu)
                    print(count)
                    for cols in sentence:
                        f.write(cols + '\t')
                    f.write('\n')
            except:
                print(docu, 'can only join an iterable')


# 通过word2vec生成嵌入特征
def word_to_vec(nodes,merchant_id):
    with open('data/'+merchant_id+'_sentence.txt','r',encoding="utf-8") as f:
        sentences = []
        for line in f:
            cols = line.strip().split('\t')
            sentences.append(cols)


    # 通过模型提取出300个特征
    model = gensim.models.Word2Vec(sentences, sg=1, size=200, alpha=0.025, window=3, min_count=1, max_vocab_size=None, sample=1e-3, seed=1, workers=45, min_alpha=0.0001, hs=0, negative=20, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=1e4)

    outfile = 'model/'
    fname = merchant_id+'_embedding'
    model.save(outfile+fname)
    model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
    # 将特征保存
    model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

def get_embeddings(live_addr,merchant_id):
    fname = 'model/'+merchant_id + '_embedding'
    model = gensim.models.Word2Vec.load(fname)

    a = model.most_similar('上海市')
    a = model.most_similar('甘肃省')
    print(a)

    # print(model.wv['甘肃省'])
    # print(model.wv['陕西省'])


    count = 0
    features = {}
    node = set()
    for docu in live_addr:
        try:
            if docu:
                count += 1

                # 分词
                sentence = jieba_split(docu)
                embd = []
                for word in sentence:
                    embd.append(model.wv[word])
                node.add(docu)
                features[docu] = embd

            if count == 5000:
                break
        except Exception as e:
            print(docu, e)

    return features,node

# 将embedding向量转化为二维空间数据
def plot_embeddings(embeddings, nodes):
    # 读取node id 和对应标签
    X = nodes

    emb_list = []
    for k in X:
        emb_list.append(embeddings[str(k)])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault('0', [])
        color_idx['0'].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    merchant_id = '3410'
    data = pd.read_csv('../LDA/addr/3410/3410addr_label.csv')

    # 删除任何一行有空值的记录
    data.dropna(axis=0, how='any', inplace=True)

    live_addr = data['addr']

    # 1 生成词语句
    # write_sentence(live_addr,merchant_id)
    # 训练模型
    # word_to_vec(None, merchant_id)

    # 获得地址的嵌入向量
    features,node = get_embeddings(live_addr,merchant_id)

    plot_embeddings(features, list(node))