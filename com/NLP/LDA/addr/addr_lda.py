# _*_coding:utf-8 _*_
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
import jieba
import numpy as np
import pandas as pd
import re
from com.NLP.jieba import jieba_test
import xlsxwriter

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(''.join(re.findall('[\u4e00-\u9fa5]+', text)))
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)

    return seg_list

def addr_to_dict(live_addr):
    doc_complete = []
    count = 0
    for app in live_addr:
        try:
            if app:
                count += 1
                # app = eval(app)
                # 去除含有.的数据项
                # app = [a for a in app if '.' not in a]
                temp = ''
                # app = app.reverse()
                app = temp.join(app)
                # print(app)

                # 分词
                doc_complete.append(jieba_split(app))
        except:
            print(app,'can only join an iterable')

        # if count == 1000:
        #     break

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(doc_complete)

    # dictionary.save('3410_201905/dict/miaola.dict')
    print('dict len', len(dictionary))

    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    # 即给每一个词一个编号，并且给出这个词的词频
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    print('dict len', len(dictionary), 'DT len', len(doc_term_matrix))

    # 使用 gensim 来创建 LDA 模型对象
    Lda = models.ldamodel.LdaModel

    for i in [30,40,60,70,90]:
        # 在 DT 矩阵上运行和训练 LDA 模型
        ldamodel = Lda(doc_term_matrix, num_topics=i, id2word=dictionary)

        # 主题一致性计算
        cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='u_mass', dictionary=dictionary,
                            processes=-1)
        # cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='c_v', dictionary=dictionary,
        #                     processes=-1)

        # ldamodel.save('3410_201905/addr_lda' + str(i) + '.model')
        # ldamodel.save('3410/addr_lda' + str(i) + '.model')
        ldamodel.save('model/addr_lda' + str(i) + '.model')

        print('u_mass_valus is: %f' % cm.get_coherence())

# 对文本进行主题预测，将预测结果以概率形式输出，作为特征数据
def get_all_topic(data, model, dictionary,t_num):
    """get data topic
    :param model: LDA model
    :param dictionary: LDA dictionary document
    :param data: string
    :return :a dic of topics and its prob
    """

    test_cut = list(jieba.cut(''.join(re.findall('[\u4e00-\u9fa5]+', data))))
    doc_bow = dictionary.doc2bow(test_cut)
    topics = model.get_document_topics(doc_bow)
    #top_id = [i[0] for i in topics]
    top_list = [0]*t_num
    for i in topics:
        top_list[i[0]] = i[1]
    return (top_list)

def dict_to_lda(live_addr):
    count = 0
    dictionary = corpora.Dictionary.load("3410_201905/dict/miaola.dict")
    # dictionary = corpora.Dictionary.load("3410/dict/miaola.dict")
    topic_num = 65

    # 载入模型文件
    # lda = models.LdaModel.load('3410_201905/addr_lda'+str(topic_num)+'.model')
    lda = models.LdaModel.load('3410/addr_lda' + str(topic_num) + '.model')

    print("准备写表格")
    f = xlsxwriter.Workbook('3410_201905/add_topics'+str(topic_num)+'_mh_dl.xlsx')
    # f = xlsxwriter.Workbook('3410/add_topics' + str(topic_num) + '.xlsx')
    sheet1 = f.add_worksheet(u'sheet1')
    s = 0
    file_dict = ['tp'+str(i) for i in range(topic_num)]
    for key in file_dict:
        sheet1.write(0, s, key)
        s += 1
    print("表头设置完成")

    for app in live_addr:
        try:
            count+=1
            # app = eval(app)
            # 去除含有.的数据项
            # app = [a for a in app if '.' not in a]
            temp = ''
            app = temp.join(app)

            # 获得预测题主概率分布
            tps = get_all_topic(app, lda, dictionary,topic_num)
            c = 0
            for t in tps:
                sheet1.write(count, c, t)
                c += 1

            print(count)
        except:
            print('error')

    f.close()  # 保存文件

def get_top_topic():
    # 载入模型文件
    lda = models.LdaModel.load('model/addr_lda' + str(50) + '.model')

    print(lda.print_topics(14))
    print(lda.print_topics(19))
    print(lda.print_topics(4))

def perplexity_check(app_list,topic):
    doc_complete = []
    count = 0
    for app in app_list:
        try:
            count += 1
            # app = eval(app)
            # 去除含有.的数据项
            # app = [a for a in app if '.' not in a]
            temp = ''
            app = temp.join(app)
            # print(app)

            # 分词
            doc_complete.append(jieba_split(app))
        except:
            print(app,'can join ')

    # dictionary = corpora.Dictionary.load("model/dict/miaola.dict")
    dictionary = corpora.Dictionary(doc_complete)

    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    perplexity_list = []
    for i in topic:
        # ldamodel = models.LdaModel.load('3410/addr_lda' + str(i) + '.model')
        # ldamodel = models.LdaModel.load('3410_201905/addr_lda' + str(i) + '.model')
        ldamodel = models.LdaModel.load('model/addr_lda' + str(i) + '.model')
        perp = perplexity(ldamodel, doc_term_matrix, dictionary, len(dictionary.keys()), num_topics=i)
        print(i,perp)
        perplexity_list.append(perp)

    return perplexity_list

import math
#计算困惑度，评价模型
def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    # print('the info of this ldamodel: \n')
    # print('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        try:
            doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
        except:
            print('out of bounds')
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for ddd in doc:
            word_id = ddd[0]
            num = ddd[1]
            prob_word = 0.0 # the probablity of the word
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    # print("the perplexity of this ldamodel is : %s"%prep)
    return prep

import matplotlib.pyplot as plt
def graph_draw(topic,perplexity):             #做主题数与困惑度的折线图
     x=topic
     y=perplexity
     plt.plot(x,y,color="red",linewidth=2)
     plt.xlabel("Number of Topic")
     plt.ylabel("Perplexity")
     plt.show()

if __name__ == '__main__':
    import time

    starttime = time.time()


    data = pd.read_csv('model/addr_label.csv')
    # data = pd.read_csv('3410/3410addr_label.csv')
    # data = pd.read_csv('3410_201905/3410addr_label_201905.csv')

    # 删除任何一行有空值的记录
    data.dropna(axis=0, how='any', inplace=True)

    live_addr = data['addr']
    #
    # # 训练模型
    # addr_to_dict(live_addr)
    #
    # # 预测主题
    # dict_to_lda(live_addr)

    # get_top_topic()

    topic = [30,40,50, 60,70, 80,90,100]
    # topic = [55,65,70,75]

    # 困惑度
    perplexity_list = perplexity_check(live_addr,topic)

    # perplexity_list = [593.87015,534.20089,520.11836,499.83862,496.16712,504.97288,518.06380,563.74753]

    # 困惑度走势
    graph_draw(topic, perplexity_list)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)