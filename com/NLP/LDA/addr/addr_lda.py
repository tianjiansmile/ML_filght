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
    seg_list = jieba.cut(text, cut_all=False)
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)

    return seg_list

def addr_to_dict(live_addr):
    doc_complete = []
    count = 0
    for app in live_addr:
        try:
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

    dictionary.save('model/dict/miaola.dict')
    print('dict len', len(dictionary))

    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    # 即给每一个词一个编号，并且给出这个词的词频
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    print('dict len', len(dictionary), 'DT len', len(doc_term_matrix))

    # 使用 gensim 来创建 LDA 模型对象
    Lda = models.ldamodel.LdaModel

    for i in [50, 80,150]:
        # 在 DT 矩阵上运行和训练 LDA 模型
        ldamodel = Lda(doc_term_matrix, num_topics=i, id2word=dictionary)

        # 主题一致性计算
        cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='u_mass', dictionary=dictionary,
                            processes=-1)
        # cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='c_v', dictionary=dictionary,
        #                     processes=-1)

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
    dictionary = corpora.Dictionary.load("model/dict/miaola.dict")
    topic_num = 50

    # 载入模型文件
    lda = models.LdaModel.load('model/addr_lda'+str(topic_num)+'.model')

    print("准备写表格")
    f = xlsxwriter.Workbook('add_topics'+str(topic_num)+'.xlsx')
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

if __name__ == '__main__':
    import time

    starttime = time.time()

    data = pd.read_excel('秒啦首贷_train.xlsx',sheetname='Sheet1')
    # data = pd.read_csv('miaola_extact_5000.csv')
    live_addr = data['live_addr']
    #
    # # 训练模型
    # addr_to_dict(live_addr)
    #
    # # 预测主题
    dict_to_lda(live_addr)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)