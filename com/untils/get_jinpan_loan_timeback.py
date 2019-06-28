import xlwt
import json
import time
import requests
import queue as Queue
import threading

# 本脚本主要通过用户身份证号，已经回溯时间获取用户历史借贷特征

# 设置队列长度
workQueue = Queue.Queue(40000)

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
threadList = ["Thread-1", "Thread-2"]
# 线程池
threads = []

def crawler(threadName, q,wf):
    # 从队列里获取url
    idNum_Date = q.get(timeout=2)
    idNum=idNum_Date[0]
    curr_date=idNum_Date[1]
    try:
        all_list = get_response(idNum,curr_date)
        if all_list:
            print('获取到特征值，开始写入到文本')
            wf.write(str(all_list)+'\n')
            wf.flush()
        # 打印：队列长度，线程名，响应吗，正在访问的url
        print(q.qsize(), threadName, idNum)
    except Exception as e:
        print(q.qsize(), threadName, "Error: ", e)

# 获取test.txt中所有身份证的107条规则并写入文本
def get_value(file):
    file_idNum=open('huaxiatest.txt',encoding='utf-8').readlines()
    for i in file_idNum:
        # 去掉结尾空格
        i=i.rstrip()
        # 取出身份证号码
        # order_id = i.split(',')[0]
        idNum=i.split(',')[0]
        curr_date = i.split(',')[1]
        if not idNum:
            break
        workQueue.put([idNum,curr_date])
    # 创建新线程
    with open(file,'w') as wf:
        for tName in threadList:
            thread = myThread(tName, workQueue,wf)
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for t in threads:
            t.join()

        # all_list=get_response(idNum)
        # if all_list:
        #     print('获取到%s特征值，开始写入到文本'%idNum)
        #     file3.write(str(all_list)+'\n')
        #     file3.flush()
    print('写入完成')
    time.sleep(2)

# 根据身份证号码获取107条特征，并转换成list输出
def get_response(idNum,currDate):
    all_dic={}
    url='http://127.0.0.1:5000/getEncyUserFeaturesTest?identityNo=%s&currDate=%s'
    res=requests.get(url % (idNum,currDate))
    if res.status_code==200:
        all_list=[]
        res=res.json()
        result=res.get('result')
        all_dic['identity_no']=result.get('identity_no')
        features=result.get('features')
        if features:
            all_dic.update(features)
        for key in all_dic:
            all_list.append(all_dic[key])
        return all_list
    else:
        return None

# 将结果写入表格
def wirte_excle():
    file = 'value.txt'
    get_value(file)
    print("准备写表格")
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    # #将列名写入第0行
    file_dict = {'idNum': '', "apply_pdl_7": 0, "apply_int_7": 0, "apply_sum_7": 0, "reject_pdl_7": 0,
                 "reject_int_7": 0, "reject_sum_7": 0, "approve_pdl_7": 0, "approve_int_7": 0, "approve_sum_7": 0,
                 "overdue_pdl_7": 0, "overdue_int_7": 0, "overdue_sum_7": 0, "loanamount_pdl_7": 0.0,
                 "loanamount_int_7": 0.0, "loanamount_sum_7": 0.0, "maxOverdue_pdl_7": -99999,
                 "maxOverdue_int_7": -99999, "maxOverdue_sum_7": -99999, "apply_pdl_14": 0, "apply_int_14": 0,
                 "apply_sum_14": 0, "reject_pdl_14": 0, "reject_int_14": 0, "reject_sum_14": 0, "approve_pdl_14": 0,
                 "approve_int_14": 0, "approve_sum_14": 0, "overdue_pdl_14": 0, "overdue_int_14": 0,
                 "overdue_sum_14": 0, "loanamount_pdl_14": 0.0, "loanamount_int_14": 0.0, "loanamount_sum_14": 0.0,
                 "maxOverdue_pdl_14": -99999, "maxOverdue_int_14": -99999, "maxOverdue_sum_14": -99999,
                 "apply_pdl_30": 0, "apply_int_30": 0, "apply_sum_30": 0, "reject_pdl_30": 0, "reject_int_30": 0,
                 "reject_sum_30": 0, "approve_pdl_30": 0, "approve_int_30": 0, "approve_sum_30": 0, "overdue_pdl_30": 0,
                 "overdue_int_30": 0, "overdue_sum_30": 0, "loanamount_pdl_30": 0.0, "loanamount_int_30": 0.0,
                 "loanamount_sum_30": 0.0, "maxOverdue_pdl_30": -99999, "maxOverdue_int_30": -99999,
                 "maxOverdue_sum_30": -99999, "apply_pdl_60": 0, "apply_int_60": 0, "apply_sum_60": 0,
                 "reject_pdl_60": 0, "reject_int_60": 0, "reject_sum_60": 0, "approve_pdl_60": 0, "approve_int_60": 0,
                 "approve_sum_60": 0, "overdue_pdl_60": 0, "overdue_int_60": 0, "overdue_sum_60": 0,
                 "loanamount_pdl_60": 0.0, "loanamount_int_60": 0.0, "loanamount_sum_60": 0.0,
                 "maxOverdue_pdl_60": -99999, "maxOverdue_int_60": -99999, "maxOverdue_sum_60": -99999,
                 "apply_pdl_90": 0, "apply_int_90": 0, "apply_sum_90": 0, "reject_pdl_90": 0, "reject_int_90": 0,
                 "reject_sum_90": 0, "approve_pdl_90": 0, "approve_int_90": 0, "approve_sum_90": 0, "overdue_pdl_90": 1,
                 "overdue_int_90": 0, "overdue_sum_90": 0, "loanamount_pdl_90": 0.0, "loanamount_int_90": 0.0,
                 "loanamount_sum_90": 0.0, "maxOverdue_pdl_90": 0, "maxOverdue_int_90": -99999,
                 "maxOverdue_sum_90": 0, "apply_pdl_180": 0, "apply_int_180": 0, "apply_sum_180": 0,
                 "reject_pdl_180": 0, "reject_int_180": 0, "reject_sum_180": 0, "approve_pdl_180": 0,
                 "approve_int_180": 0, "approve_sum_180": 0, "overdue_pdl_180": 0, "overdue_int_180": 0,
                 "overdue_sum_180": 0, "loanamount_pdl_180": 0.0, "loanamount_int_180": 0.0,
                 "loanamount_sum_180": 0.0, "maxOverdue_pdl_180": 0, "maxOverdue_int_180": -99999,
                 "maxOverdue_sum_180": 0, "apply_pdl_all": 0, "apply_int_all": 0, "apply_sum_all": 0,
                 "reject_pdl_all": 0, "reject_int_all": 0, "reject_sum_all": 0, "approve_pdl_all": 0,
                 "approve_int_all": 0, "approve_sum_all": 0, "overdue_pdl_all": 0, "overdue_int_all": 0,
                 "overdue_sum_all": 0, "loanamount_pdl_all": 0.0, "loanamount_int_all": 0.0,
                 "loanamount_sum_all": 0.0, "maxOverdue_pdl_all": 0, "maxOverdue_int_all": -99999,"maxOverdue_sum_all": 0,
                 "apply_pdl_label": 0, "apply_int_label": 0, "apply_sum_label": 0,
                 "reject_pdl_label": 0, "reject_int_label": 0, "reject_sum_label": 0, "approve_pdl_label": 0,
                 "approve_int_label": 0, "approve_sum_label": 0, "overdue_pdl_label": 0, "overdue_int_label": 0,
                 "overdue_sum_label": 0, "loanamount_pdl_label": 0.0, "loanamount_int_label": 0.0,
                 "loanamount_sum_label": 0.0, "maxOverdue_pdl_label": 0, "maxOverdue_int_label": -99999,
                 "maxOverdue_sum_label": 0
                 }

    s=0
    for key in file_dict:
        sheet1.write(0, s, key)
        s+=1
    print("表头设置完成")

    #然后将数据写入第 i 行，第 j 列
    i = 1
    file=open(file,encoding='utf-8').readlines()
    for data in file:
        data=eval(data)
        for j in range(len(data)):
            sheet1.write(i, j, data[j])
        i = i + 1
        print("正在写入第%s行"%i)
    print("内容写入完成")
    f.save('a_new.xls')  # 保存文件



if __name__ == '__main__':
    time1 = time.time()
    wirte_excle()
    # wirte_excle1()
    time2 = time.time()
    print(time2-time1)