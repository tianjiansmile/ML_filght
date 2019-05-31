# _*_coding:utf-8 _*_
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
import jieba
import numpy as np
import pandas as pd
import re
from com.NLP.jieba import jieba_test
import xlsxwriter


def app_to_dict(app_list):
    doc_complete = []
    count= 0
    for app in app_list:
        count+=1
        app = eval(app)
        # 去除含有.的数据项
        app = [a for a in app if '.' not in a]
        temp = ''
        app = temp.join(app)
        # print(app)

        # 分词
        doc_complete.append(jieba_split(app))

        # if count == 1000:
        #     break

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(doc_complete)

    dictionary.save('model/dict/miaola_8w.dict')
    print('dict len',len(dictionary))

    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    # 即给每一个词一个编号，并且给出这个词的词频
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_complete]

    print('dict len', len(dictionary),'DT len',len(doc_term_matrix))

    # 使用 gensim 来创建 LDA 模型对象
    Lda = models.ldamodel.LdaModel

    for i in [280,350,400]:
        # 在 DT 矩阵上运行和训练 LDA 模型
        ldamodel = Lda(doc_term_matrix, num_topics=i, id2word=dictionary)


        # 主题一致性计算
        cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='u_mass', dictionary=dictionary, processes=-1)
        # cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='c_v', dictionary=dictionary,
        #                     processes=-1)

        ldamodel.save('model/miaola_lda'+str(i)+'.model')

        print('u_mass_valus is: %f' % cm.get_coherence())


def dict_to_lda(app_list):
    count = 0
    dictionary = corpora.Dictionary.load("model/dict/miaola_8w.dict")
    topic_num = 350

    # 载入模型文件
    lda = models.LdaModel.load('model/miaola_lda350.model')

    print("准备写表格")
    f = xlsxwriter.Workbook('topics350.xlsx')
    sheet1 = f.add_worksheet(u'sheet1')
    s = 0
    file_dict = ['tp'+str(i) for i in range(topic_num)]
    for key in file_dict:
        sheet1.write(0, s, key)
        s += 1
    print("表头设置完成")

    for app in app_list:
        count+=1
        app = eval(app)
        # 去除含有.的数据项
        app = [a for a in app if '.' not in a]
        temp = ''
        app = temp.join(app)

        # 获得预测题主概率分布
        tps = get_all_topic(app, lda, dictionary,topic_num)
        c = 0
        for t in tps:
            sheet1.write(count, c, t)
            c += 1

        print(count)

    f.close()  # 保存文件


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

def jieba_split(text):
    # 精确模式
    seg_list = jieba.cut(text, cut_all=False)
    # print(u"[精确模式]: ", "/ ".join(seg_list))

    seg_list = list(seg_list)

    return seg_list

def cut_csv():
    data = pd.read_csv('miaola_extact.csv')
    data = data[:5000]

    data.to_csv('miaola_extact_5000.csv')
    print(data.shape)

# 通过app数据获得预测主题概率
def get_text_topic(text):
    dictionary = corpora.Dictionary.load("model/dict/miaola_5k.dict")
    topic_num = 120

    # 载入模型文件
    lda = models.LdaModel.load('model/miaola_lda120.model')

    app = eval(text)
    # 去除含有.的数据项
    app = [a for a in app if '.' not in a]
    temp = ''
    app = temp.join(app)

    tps = get_all_topic(app, lda, dictionary, topic_num)

    print(tps)


if __name__ == '__main__':
    import time

    starttime = time.time()

    # cut_csv()

    # text = "['云文件夹', 'com.android.cts.priv.ctsshim', '相机', '玩机技巧', 'Android Services Library', 'HwSynergy', '华为桌面', 'com.android.gallery3d.overlay', '音乐', '通话/信息存储', '换机助手', 'UEInfoCheck', '录音机', '日历存储', '文件管理', 'hiview', 'com.huawei.iaware', '通过蓝牙导入', '媒体存储', '王者荣耀', '主题', 'HwChrService', '华为服务框架', '爱奇艺', 'Google One Time Init', 'Android Shared Library', 'com.android.wallpapercropper', 'HwNearby', '悬浮导航', '中国农业银行', '美团', '智能助手', '手势服务', 'SmartcardService', '华为应用市场', 'HwIntelligentRecSystem', '学生模式', '伟文标记', '11选5助手', 'Huawei Secure IME', '文件', '外部存储设备', 'HTML 查看器', 'Companion Device Manager', 'RegService', '讯飞语音引擎', 'MmsService', '华为钱包', '天气数据服务', '下载管理器', '游戏助手', '美图秀秀', '营装维服一体化平台', 'com.huawei.cryptosms.service.CryptoMessageApplication', '查找我的手机', '设备认证服务', '华为连接服务', '会员服务', '屏幕录制', '视频编辑', '天天中彩票', '隐私空间', '华为浏览器', 'ConfigUpdater', '录音机', '元贝驾考', 'iConnect', 'AutoRegSms', '备份', '视频•优酷版', '威龙在线', '软件包访问帮助程序', 'com.huawei.hiviewtunnel', '下载内容', 'Google Play 服务更新程序', 'PacProcessor', '智能家居', 'androidhwext', '微信', '百度输入法华为版', 'com.android.frameworkhwext.honor', '证书安装器', 'huawei.android.widget', 'com.android.carrierconfig', 'TalkBack', 'Android 系统', '花粉俱乐部', '华为移动服务', '智能提醒', '联系人', '取词', 'com.huawei.systemserver.HwServerApplication', 'com.android.frameworkhwext.dark', '信息', 'MTP 服务', 'NFC 服务', 'SIM 卡应用', '水印相机', 'com.android.backupconfirm', 'Huawei Share', 'HwIndexSearchObserverService', '智能解锁', '支付保护中心', 'Intent Filter Verification Service', 'HwIndexSearchService', 'Huawei Share', '华为游戏中心', 'FIDO UAF ASM', '日历', 'HwWifiproBqeService', '设置存储', 'com.android.sharedstoragebackup', '搜狗手机助手（水印相机）', '打印处理服务', '手机管家', 'com.android.frameworkres.overlay', '基本互动屏保', '拨号', '手机管家', '腾讯视频', '输入设备', '在线黄页', 'QQ浏览器', '默认打印服务', '虾米音乐', '手机克隆', 'HiLink', 'Android System WebView', '滚动截屏', '决策系统', '语音助手', '通话管理', 'Google通讯录同步', '备忘录', '密钥链', '华为杂志锁屏', 'UC浏览器', '图库', '手表应用同步', 'Google Play 服务', 'Google服务框架', 'HwStartupGuide', 'Call Log Backup/Restore', 'Google合作伙伴设置', 'QQ', 'SJ', 'FIDO UAF Client', '动态巡检', '打包安装程序', 'Pico TTS', '华为阅读', 'ProxyHandler', '无线分享', '扫名片', '豌豆荚', 'Print Service Recommendation Service', '百度极速版', '工作资料设置', '天气小工具', '华为视频', '智能识屏', '指南针', '图片屏保程序', 'CAD快速看图', '百度地图', '双卡管理', 'HwAps', '系统更新', '华为钱包安全支付', 'WLAN 直连', 'Live Wallpaper Picker', '高德位置服务', '浦发信用卡', 'MMITest', '标记', '中国建设银行', '省电精灵', '139邮箱', '智能截屏', '掌上运维', 'Google 备份传输', '融360', 'HwInstantOnline', '存储空间管理器', '设置', '神州专车', '华为商城', '计算器', '高德地图', '应用商店（美图秀秀）', '工程菜单', '天气', '携程旅行', '和家亲', 'com.android.cts.ctsshim', '智能遥控', '掌上代维管理系统', '推送服务', 'VpnDialogs', '资源采集', '电子邮件', '奇迹单机版', '拨号', 'Shell', 'com.android.wallpaperbackup', '存储已屏蔽的号码', '用户词典', '用钱宝', '个人紧急信息', '扫一扫', '铁通生产安全', '融合定位', '时钟', '系统用户界面', 'Exchange 服务', 'Bluetooth MIDI Service', '集客应用', '西瓜视频', '智能检测', 'K 歌特效', '手机淘宝', 'HwUE', 'CAService', '天际通数据服务', 'HwImsService', 'HwLBSService', '语义分析', '蓝牙', '云南移动', '联系人存储', 'CaptivePortalLogin', 'HiAction', '摩登娱乐', '镜子', '支付宝', '多屏互动', 'WPS Office', '元道经纬相机', '华为 RCS 服务']"
    # get_text_topic(text)

    # jieba_split('精品推荐com.android.cts.priv.ctsshim相机玩机技巧Android Services LibraryHwSynergy华为桌面音乐通话/信息存储UEInfoCheck录音机日历存储文件管理hiviewcom.huawei.iaware')

    data = pd.read_csv('miaola_extact.csv')
    # data = pd.read_csv('miaola_extact_5000.csv')
    app_list = data['app_list']
    #
    # # 训练模型
    # app_to_dict(app_list)
    #
    # # 预测主题
    dict_to_lda(app_list)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)