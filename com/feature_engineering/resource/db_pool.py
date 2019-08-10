import pymysql
from DBUtils.PooledDB import PooledDB, SharedDBConnection
from DBUtils.PersistentDB import PersistentDB, PersistentDBError, NotSupportedError
from com.risk_score.feature_extact import setting
config = {
    'host': setting.tidb_host,
    'port': 3306,
    'database': setting.tidb_db,
    'user': setting.tidb_user,
    'password': setting.tidb_pwd,
    'charset': 'utf8'
}

# config = {
#     'host': CONFIG['mycat.host'],
#     'port': 8066,
#     'database': 'user_db',
#     'user': CONFIG['mycat.user'],
#     'password': CONFIG['mycat.pwd'],
#     'charset': 'utf8'
# }


def get_db_pool(is_mult_thread):
    if is_mult_thread:
        poolDB = PooledDB(
            # 指定数据库连接驱动
            creator=pymysql,
            # 连接池允许的最大连接数,0和None表示没有限制
            maxconnections=2000,
            # 初始化时,连接池至少创建的空闲连接,0表示不创建
            mincached=50,
            # 连接池中空闲的最多连接数,0和None表示没有限制
            maxcached=0,
            # 连接池中最多共享的连接数量,0和None表示全部共享(其实没什么卵用)
            maxshared=0,
            # 连接池中如果没有可用共享连接后,是否阻塞等待,True表示等等,
            # False表示不等待然后报错
            blocking=True,
            # 开始会话前执行的命令列表
            setsession=[],
            # ping Mysql服务器检查服务是否可用
            ping=0,
            **config
        )
    else:
        poolDB = PersistentDB(
            # 指定数据库连接驱动
            creator=pymysql,
            # 一个连接最大复用次数,0或者None表示没有限制,默认为0
            maxusage=1000,
            **config
        )
    return poolDB

