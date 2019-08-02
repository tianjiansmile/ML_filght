import xlwt
import json
import time
import requests
import queue as Queue
import threading
from com.feature_engineering.resource.dict import resource_dict

# 本脚本主要通过用户身份证号或是order_id，已经回溯时间获取用户历史资源数据特征

# 设置队列长度
workQueue = Queue.Queue(500000)

class myThread(threading.Thread):
    def __init__(self, name, q,wf):
        threading.Thread.__init__(self)
        self.name = name
        self.q = q
        self.wf = wf

    def run(self):
        print("Starting " + self.name)
        while True:
            try:
                crawler(self.name, self.q,self.wf)
            except:
                break
        print("Exiting " + self.name)

# 创建5个线程名
threadList = ["Thread-1", "Thread-2","Thread-3", "Thread-4","Thread-5", "Thread-6"
    ,"Thread-7", "Thread-8","Thread-9", "Thread-10"]
# 线程池
threads = []

def crawler(threadName, q,wf):
    # 从队列里获取url
    idNum_Date = q.get(timeout=2)
    idNum=idNum_Date[0]
    curr_date = idNum_Date[1]
    overdue=idNum_Date[2]

    try:
        all_list = get_response(idNum,curr_date,overdue)
        if all_list:
            print('获取到特征值，开始写入到文本')
            wf.write(str(all_list)+'\n')
            wf.flush()
        # 打印：队列长度，线程名，响应吗，正在访问的url
        print(q.qsize(), threadName, idNum)
    except Exception as e:
        print(q.qsize(), threadName, "Error: ", e)

# 获取test.txt中所有身份证的107条规则并写入文本
def get_value(file,loan):
    conn = loan.connection_jinpan()
    file_idNum=open('data/好贷借.csv',encoding='GBK').readlines()
    for i in file_idNum:
        # 去掉结尾空格
        i=i.rstrip()
        # 取出身份证号码
        # order_id = i.split(',')[0]
        create_time =i.split(',')[0]
        identity_no = i.split(',')[1]
        overdue = i.split(',')[2]
        # result = loan.get_order_user(order_id, conn)
        # if result:
        #     identity_no = result[1]
        #     if not identity_no:
        #         continue
        workQueue.put([identity_no,create_time,overdue])
    # 创建新线程
    with open(file,'w') as wf:
        for tName in threadList:
            thread = myThread(tName, workQueue,wf)
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for t in threads:
            t.join()

        # all_list=get_response(identity_no)
        # if all_list:
        #     print('获取到%s特征值，开始写入到文本'%identity_no)
        #     file.write(str(all_list)+'\n')
        #     file.flush()
    print('写入完成')
    time.sleep(2)

# 根据身份证号码获取107条特征，并转换成list输出
def get_response(idNum,currDate,overdue):
    all_dic={}
    url='http://127.0.0.1:5000/getJinpanResFeatures?identityNo=%s&create_time=%s'
    res=requests.get(url % (idNum,currDate))
    if res.status_code==200:
        all_list=[]
        res=res.json()
        result=res.get('result')
        # all_list.append(idNum)
        if result:
            all_dic.update(result)
        for key in all_dic:
            all_list.append(all_dic[key])

        all_list.append(overdue)
        return all_list
    else:
        return None

# 将结果写入表格
def wirte_excle(loan):
    file = 'data/haojiedai.txt'
    get_value(file,loan)
    # print("准备写表格")
    # f = xlwt.Workbook()
    # sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    # # #将列名写入第0行
    # file_dict = resource_dict.res_dict
    #
    # s=0
    # for key in file_dict:
    #     sheet1.write(0, s, key)
    #     s+=1
    # print("表头设置完成")
    #
    # #然后将数据写入第 i 行，第 j 列
    # i = 1
    # file=open(file,encoding='utf-8').readlines()
    # for data in file:
    #     data=eval(data)
    #     for j in range(len(data)):
    #         sheet1.write(i, j, data[j])
    #     i = i + 1
    #     print("正在写入第%s行"%i)
    # print("内容写入完成")
    # f.save('data/a_new.xls')  # 保存文件



if __name__ == '__main__':
    from com.risk_score.feature_extact import setting
    from com.feature_engineering.resource.resource_data_analysis import LoanHistoryResourceFeatureGroup

    loan = LoanHistoryResourceFeatureGroup(setting)
    time1 = time.time()
    wirte_excle(loan)
    # wirte_excle1()
    time2 = time.time()
    print(time2-time1)