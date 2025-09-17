# TensorFlow实现卷积神经网络
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_utils
from L4W1.cnn_utils import convert_to_one_hot

# np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()

# plt.imshow(X_train_orig[6])
# plt.show()
# print(np.squeeze(Y_train_orig[:, 6]))

X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# print(X_train.shape)   # (1080, 64, 64, 3)
# print(Y_train.shape)   # (1080, 6)
# print(X_test.shape)    # (120, 64, 64, 3)
# print(Y_test.shape)    # (120, 6)

# 初始化参数
def init_parameters():
    initializer = tf.keras.initializers.GlorotUniform(seed=0)
    W1 = tf.Variable(initializer(shape=[4, 4, 3, 8]), name="W1", trainable=True, dtype=tf.float32)
    W2 = tf.Variable(initializer(shape=[2, 2, 8, 16]), name="W2", trainable=True, dtype=tf.float32)

    parameters = {"W1": W1, "W2": W2}
    return parameters

# parameters = init_parameters()
# print(parameters['W1'].numpy().flatten()[:2])
# print(parameters['W2'].numpy().flatten()[:2])

# 正向传播

# 确保参数可追踪
flatten_layer = tf.keras.layers.Flatten()
dense_layer = tf.keras.layers.Dense(6, activation=None)

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool2d(A1, ksize=8, strides=8, padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool2d(A2, ksize=4, strides=4, padding='SAME')

    P2_flat = flatten_layer(P2)
    Z3 = dense_layer(P2_flat)

    return Z3

# 计算损失
def compute_cost(Z3, Y):
    Z3 = tf.cast(Z3, tf.float32)
    Y = tf.cast(Y, tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost

# 构建模型
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True):

    tf.random.set_seed(1)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    costs = []
    seed = 3

    parameters = init_parameters()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    for epoch in range(num_epochs):
        minibatch_cost = 0
        num_minibatches = int(m / minibatch_size)
        seed = seed + 1
        minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            with tf.GradientTape() as tape:
                # 正向传播
                Z3 = forward_propagation(minibatch_X, parameters)
                # 计算损失
                loss = compute_cost(Z3, minibatch_Y)

            # parameters中不包含全连接层的参数，需要另外加入
            all_vars = list(parameters.values()) + dense_layer.trainable_variables
            # 计算梯度
            gradients = tape.gradient(loss, all_vars)
            # 参数更新
            optimizer.apply_gradients(zip(gradients, all_vars))

            minibatch_cost += loss / num_minibatches

        if print_cost and epoch % 5 == 0:
            print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost:
            costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    Z3_train = forward_propagation(X_train, parameters)  # shape: (m, n_y)
    pred_train = tf.argmax(Z3_train, axis=1)  # 每行最大值索引 → (m,)
    true_train = tf.argmax(Y_train, axis=1)  # (m,)
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_train, true_train), tf.float32)).numpy()

    Z3_test = forward_propagation(X_test, parameters)
    pred_test = tf.argmax(Z3_test, axis=1)
    true_test = tf.argmax(Y_test, axis=1)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_test, true_test), tf.float32)).numpy()

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return parameters

parameters = model(X_train, Y_train, X_test, Y_test)