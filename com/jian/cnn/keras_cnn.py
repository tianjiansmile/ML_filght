import tensorflow as tf
from tensorflow import keras
from tensorflow import layers

# https://www.jianshu.com/p/d02980fd7b54

# 保存模型的权重时，tf.keras默认为 checkpoint 格式。 通过save_format ='h5'使用HDF5。
# 典型的CNN架构


from keras.datasets import mnist
from keras.utils import to_categorical

train_X, train_y = mnist.load_data()[0]
train_X = train_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')
train_X /= 255
train_y = to_categorical(train_y, 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

model = Sequential()
# 第一层：卷积层
# 这一层的输入的原始的图像像素，该模型接受的图像为28*28*1，32个5*5卷积核，步长为1，不使用全0填充。
# 所以这层输出的尺寸为18-5+1=14，这个张量是长宽14*14深度为32，相当于32张卷积后的照片
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
# 再一层卷积层，该层接受的图像为14*14*32，64个5*5卷积核
# # 所以这层输出的尺寸为14-5+1=10，这个张量是长宽10*10深度为64，相当于64张卷积后的照片
model.add(Conv2D(64, (5,5), activation='relu'))
# 再一层池化层，该层接受的图像为10*10*64，输出尺度 5*5*64
model.add(MaxPool2D(pool_size=(2,2)))
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())
model.add(Dropout(0.5))
# 再一层全连接层 输出128维
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 最后一层输出层
model.add(Dense(10, activation='softmax'))
# 编译
model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])

batch_size = 100
epochs = 8
# 训练
model.fit(train_X, train_y,
         batch_size=batch_size,
         epochs=epochs)

test_X, test_y = mnist.load_data()[1]
test_X = test_X.reshape(-1, 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y = to_categorical(test_y, 10)
loss, accuracy = model.evaluate(test_X, test_y, verbose=1)
print('loss:%.4f accuracy:%.4f' %(loss, accuracy))