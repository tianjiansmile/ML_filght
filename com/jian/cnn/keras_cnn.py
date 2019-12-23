import tensorflow as tf
from tensorflow import keras
from tensorflow import layers

# https://www.jianshu.com/p/d02980fd7b54

# 保存模型的权重时，tf.keras默认为 checkpoint 格式。 通过save_format ='h5'使用HDF5。


# 1 Sequential model 构建一个简单的全连接网络（即多层感知器）
model = keras.Sequential()
# Adds a 64个神经元的全连接层:
model.add(keras.layers.Dense(64, activation='relu'))
# Add another:
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

# activation：设置层的激活函数。 此参数由内置函数的名称或可调用对象指定。 默认情况下，不应用任何激活。
# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# kernel_initializer 和 bias_initializer：设置层创建时，权重和偏差的初始化方法。指定方法：名称 或 可调用对象。默认为"Glorot uniform" initializer
# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])