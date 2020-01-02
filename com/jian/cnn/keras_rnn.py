# coding: utf-8
import numpy as np #数组模块；
from matplotlib import pyplot as plt #绘图模块；
from pandas import read_csv #导入CSV文件成dataframe结构;
import math #导入数学模块，计算均方根差使用;
from keras.models import Sequential #引入Kears模块的序列模型，此模型是将所有层线性叠加；
from keras.layers import Dense #输出层使用全连接层;
from keras.layers import LSTM #引入LSTM层;
from sklearn.preprocessing import MinMaxScaler #数据标准化
from sklearn.metrics import mean_squared_error #均方根差，矩阵计算;

seed = 7 #随机种子
batch_size = 10 #每批过神经网络的大小;
epochs = 100 #神经网络训练的轮次
filename = 'data/cases.csv' #数据文件，两列，一列是时间，另外一列是每天的进件量数据;
look_back=1 #时间窗口，步长为1，即用今天预测明天;

#此函数的目的是将输入的每日的进件量数据作为输入和输出，Y是X的下一个输出;
# 序列数据的回归问题不同与一般的分类问题， 这里每一个Y对应的X都是Y之前所有的序列数据
# 在RNN结构中，每一层的输入都是前一个序列值，每一层的输出都是后一个序列值，每一层的神经元会携带前若干个输入序列的信息
# 所以在构造训练数据时，x取当前值，y取后一个值
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)

#隐藏层4个，input_shape是输入数据格式，
# LSTM 层输入格式: 为矩阵,矩阵内容 [ samples, time steps, features ]
# samples:观测值，
# time steps:对于给定的观测值,给定变量有单独的时间步--就是时间窗口
# features:在得到观测值的时刻,观测到的单独的 measures--就是列数(属性个数) ;

def build_model():
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(1, look_back)))
    model.add(Dense(units=1)) #输出层采用全连接层;
    model.compile(loss='mean_squared_error', optimizer='adam') #损失函数是均方差，优化器是采用adam;
    return model


# 设置随机种子,目的是使得可以复现神经网络训练的结果;
np.random.seed(seed)

# 导入数据
data = read_csv(filename, usecols=[1], engine='python', skipfooter=1)
dataset = data.values.astype('float32')


# 标准化数据
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67) #训练数据
validation_size = len(dataset) - train_size #测试数据
train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

# 创建dataset，让数据产生相关性
X_train, y_train = create_dataset(train)
X_validation, y_validation = create_dataset(validation)

print(X_train.shape)
# 将输入转化成为【sample， time steps, feature]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))

# 训练模型
model = build_model()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, )

# 模型预测数据
predict_train = model.predict(X_train)
predict_validation = model.predict(X_validation)

# 反标准化数据 --- 目的是保证MSE的准确性
predict_train = scaler.inverse_transform(predict_train)
y_train = scaler.inverse_transform([y_train])
predict_validation = scaler.inverse_transform(predict_validation)
y_validation = scaler.inverse_transform([y_validation])

# 评估模型
train_score = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
#用根均方误差评估
print('Train Score: %.2f RMSE' % train_score)
validation_score = math.sqrt(mean_squared_error(y_validation[0], predict_validation[:, 0]))
print('Validatin Score: %.2f RMSE' % validation_score)

# 构建通过训练集进行预测的图表数据
predict_train_plot = np.empty_like(dataset)
predict_train_plot[:, :] = np.nan
predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

# 构建通过评估数据集进行预测的图表数据
predict_validation_plot = np.empty_like(dataset)
predict_validation_plot[:, :] = np.nan
predict_validation_plot[len(predict_train) + look_back * 2 + 1:len(dataset) - 1, :] = predict_validation

# 图表显示
dataset = scaler.inverse_transform(dataset)
# 原始数据
plt.plot(dataset, color='blue')
# 对训练数据的预测结果
plt.plot(predict_train_plot, color='green')
# 对测试数据的预测结果
plt.plot(predict_validation_plot, color='red')

# 图表显示最终真实值与预测结果拟合的非常好
plt.show()
