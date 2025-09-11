# TensorFlow
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
import tf_utils
import time

np.random.seed(1)

# y_hat = tf.Variable(36, name="y_hat")
# y = tf.Variable(39, name="y")
#
# loss = tf.Variable((y - y_hat)**2, name="loss")
#
# print(loss.numpy())

# 线性函数
def linear_function():
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')

    return tf.add(tf.matmul(W, X), b)

# print("result = " + str(linear_function()))

# Sigmoid
def sigmoid(z):
    z = tf.cast(z, tf.float32)  # tf.sigmoid只支持浮点数或复数
    return tf.sigmoid(z)

# print("sigmoid(12) = " + str(sigmoid(12)))

# cost
def cost(labels, logits):
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    cost_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return cost_tensor

# logits = np.array([0.2, 0.4, 0.7, 0.9])
# labels = np.array([0.0, 0.0, 1.0, 1.0])
# cost = cost(labels, logits)
# print("cost = " + str(cost))

# 独热编码
def ont_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    return one_hot_matrix

# labels = np.array([1, 2, 3, 0, 2, 1])
# one_hot = ont_hot_matrix(labels, 4)
# print(one_hot)

# 使用0和1初始化
def ones(shape):
    return tf.constant(np.ones(shape, dtype=np.float32))

# print(ones([3]))

# 使用TensorFlow构建神经网络
# 数据加载
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

# print(X_train_orig.shape)   # (1080, 64, 64, 3)
# print(Y_train_orig.shape)   # (1, 1080)
# plt.imshow(X_train_orig[0])
# plt.show()
# print(np.squeeze(Y_train_orig[:, 0]))

# 数据转化
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten / 255.0
X_test = X_test_flatten / 255.0

Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

# 初始化参数
def init_parameters():
    tf.random.set_seed(1)

    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    W1 = tf.Variable(initializer(shape=(25, 12288)), name='W1', trainable=True)
    b1 = tf.Variable(tf.zeros((25, 1)), name='b1', trainable=True)
    W2 = tf.Variable(initializer(shape=(12, 25)), name='W2', trainable=True)
    b2 = tf.Variable(tf.zeros((12, 1)), name='b2', trainable=True)
    W3 = tf.Variable(initializer(shape=(6, 12)), name='W3', trainable=True)
    b3 = tf.Variable(tf.zeros((6, 1)), name='b3', trainable=True)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# parameters = init_parameters()
# print(parameters)

# 正向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

# 计算损失
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

# 构建模型
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    tf.random.set_seed(1)
    seed = 3
    m = X_train.shape[1]
    costs = []

    # 初始化参数
    parameters = init_parameters()

    # Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        seed = seed + 1
        # Mini-batch
        minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            with tf.GradientTape() as tape:
                # 正向传播
                Z3 = forward_propagation(minibatch_X, parameters)
                # 计算损失
                loss = compute_cost(Z3, minibatch_Y)
            # 计算梯度
            gradients = tape.gradient(loss, list(parameters.values()))
            # 参数更新
            optimizer.apply_gradients(zip(gradients, list(parameters.values())))

            epoch_cost += loss / num_minibatches

        if print_cost and epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)

    # 绘图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    Z3_train = forward_propagation(X_train, parameters)
    pred = tf.argmax(Z3_train, axis=0)
    true = tf.argmax(Y_train, axis=0)
    train_accuracy = (pred == true).numpy().mean()
    print("Train Accuracy:", train_accuracy)

    Z3_test = forward_propagation(X_test, parameters)
    pred_test = tf.argmax(Z3_test, axis=0)
    true_test = tf.argmax(Y_test, axis=0)
    test_accuracy = (pred_test == true_test).numpy().mean()
    print("Test Accuracy:", test_accuracy)

    return parameters

# parameters = model(X_train, Y_train, X_test, Y_test)

#开始时间
start_time = time.time()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.time()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )
