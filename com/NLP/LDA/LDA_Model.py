#  coding: utf-8
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
import jieba
import json
import re

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)

    seg_list = jieba.cut(''.join(re.findall('[\u4e00-\u9fa5]+', text)))

    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)
    # print(seg_list)

    return seg_list

# 对文本进行主题预测，将预测结果以概率形式输出，作为特征数据
def get_all_topic(data, model, dictionary):
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
    top_list = [0]*60
    for i in topics:
        top_list[i[0]] = i[1]
    return (top_list)

def app_lda_train():
    doc_complete = []
    file = 'D:/特征提取/app列表/appData_list_0.txt'
    with open(file, 'r', encoding='UTF-8') as rf:
        count = 0
        lines = rf.readlines()
        for line in lines:
            count+=1
            line = eval(line)
            # app = line.split(',')
            idnum = line[0]
            apps = line[1]
            apps = apps.replace('","','')
            apps = apps.replace('["', '')
            apps = apps.replace('"]', '')
            apps = apps.replace(',', '')
            apps = apps.replace('  ', '')
            apps = apps.replace('[', '')
            apps = apps.replace(']', '')
            # apps = apps.split(',')
            # print(apps)
            # print(jieba_split(apps))
            doc_complete.append(jieba_split(apps))

            if count ==5000:
                break
            # for a in apps:
            #     count += 1
            #     print(a)

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(doc_complete)
    #
    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    # 即给每一个词一个编号，并且给出这个词的词频
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    # 使用 gensim 来创建 LDA 模型对象
    Lda = models.ldamodel.LdaModel

    # 可视化插件
    # pyLDAvis.enable_notebook()

    for i in [20, 60, 80, 120, 180, 200, 250, 400]:
        # 在 DT 矩阵上运行和训练 LDA 模型
        ldamodel = Lda(doc_term_matrix, num_topics=i, id2word=dictionary, passes=50)


        # 主题一致性计算
        cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='u_mass', dictionary=dictionary, processes=-1)
        # cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='c_v', dictionary=dictionary,
        #                     processes=-1)

        ldamodel.save('model/applist_lda'+str(i)+'.model')

        print('u_mass_valus is: %f' % cm.get_coherence())

        # pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
        # ldatopics = ldamodel.show_topics(formatted=False)


        print(ldamodel.print_topics(i))

    doc2bow = dictionary.doc2bow(
            ['王者', '荣耀', '爱奇艺', '开奖', '视频', '拍拍', '贷', '借款', '借钱', '借贷', '意见反馈', '开心', '消消', '乐掌易', '至尊版', '腾讯', '视频',
             '维信', '卡卡', '贷', '浏览器', '信而富', '快', '钱', '钱包', '公积金', '管家', '斗地主', '移动', '手机', '贷', '今日', '头条', '抖音', '短',
             '视频', '搜狐', '视频', '酷狗', '音乐', '内涵', '段子', '云闪付', '万能钥匙', '日历', '同步', '百度', '地图', '壹', '钱包', '一键', '锁屏',
             '现金', '巴士', '京东', '国美', '易卡小赢', '卡贷', '携程', '旅行', '微博', '手机', '淘宝', '支付宝', '你', '我', '贷', '借款'])

    # 得到一个主题编号和预测概率
    topics = ldamodel.get_document_topics(doc2bow)
    for t in topics:
        print(t)
        print("%s\t%f\n" % (ldamodel.print_topic(t[0]), t[1]))

    app_str = '789信用贷预约挂号网微信分身版KK部落万元户省钱快报象钱进金钱花日日红贷上钱包爱奇艺有得借钱咖搜狗输入法打包贷芝麻贷美团优酷视频天王白卡宝盈助手火山小视频' \
              '锦带花QQ音乐小宝贷闲聊借贷宝备用钱包良人贷QQ邮箱挺好借大众点评美图秀秀蜂鸟应急万宝荣一点钱包如意口袋狮子头保时借欢乐捕鱼人欢乐豆风花雪月啦啦花每日优鲜' \
              '趣步365应急B612咔叽小米铺淘奇宝猪贝贝青木易贷现金小站饿了么金腰袋美丽钱贷佩琪钱包易趣花金猪口袋微信花无忧美团外卖滴滴出行酷猫借呗应用宝盒马信用幸福花唐僧钱包' \
              '全城闪贷熊猫借呗百度富拓金融鑫泰凭证毒小虎牙'
    tps = get_all_topic(app_str, ldamodel, dictionary)
    print(tps)

    # ldamodel.save('model/applist_lda.model')

def lda_pect():
    doc_complete = []

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary.load("model/dict/applist_13w.dict")
    # dictionary.save('model/dict/applist_13w.dict')
    #
    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    # 即给每一个词一个编号，并且给出这个词的词频
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    # 载入模型文件
    ldamodel = models.LdaModel.load('model/applist_lda.model')

    app_str = '789信用贷预约挂号网微信分身版KK部落万元户省钱快报象钱进金钱花日日红贷上钱包爱奇艺有得借钱咖搜狗输入法打包贷芝麻贷美团优酷视频天王白卡宝盈助手火山小视频' \
              '锦带花QQ音乐小宝贷闲聊借贷宝备用钱包良人贷QQ邮箱挺好借大众点评美图秀秀蜂鸟应急万宝荣一点钱包如意口袋狮子头保时借欢乐捕鱼人欢乐豆风花雪月啦啦花每日优鲜' \
              '趣步365应急B612咔叽小米铺淘奇宝猪贝贝青木易贷现金小站饿了么金腰袋美丽钱贷佩琪钱包易趣花金猪口袋微信花无忧美团外卖滴滴出行酷猫借呗应用宝盒马信用幸福花唐僧钱包' \
              '全城闪贷熊猫借呗百度富拓金融鑫泰凭证毒小虎牙'
    tps = get_all_topic(app_str, ldamodel, dictionary)
    print(tps)

if __name__ == '__main__':
    import time

    starttime = time.time()

    # 主题模型训练
    # app_lda_train()

    # 主题模型预测
    lda_pect()

    endtime = time.time()
    print(' cost time: ', endtime - starttime)
