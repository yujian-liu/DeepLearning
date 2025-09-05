# 构建深层神经网络
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils

np.random.seed(1)

# 初始化参数
def init(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def init_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # W * 0.01 会发生梯度消失，导致cost稳定在0.64
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

# parameters = init_deep([5,4,3])
# print(parameters)

# 线性正向 LINEAR
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

# 正向线性激活 LINEAR -> ACTIVATION
def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z) if activation == "sigmoid" else relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# L层模型 [LINEAR -> RELU] * (L - 1) -> LINEAR -> SIGMOID
def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        caches.append(cache)

    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, 'sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

# X, parameters = testCases.L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

# 损失函数
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1, keepdims=True)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

# Y, AL = testCases.compute_cost_test_case()
# print(str(compute_cost(AL, Y)))

# 线性反向 LINEAR backward
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

# dZ, linear_cache = testCases.linear_backward_test_case()
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print("dA_prev: ", dA_prev)
# print("dW: ", dW)
# print("db: ", db)

# 反向线性激活 LINEAR -> ACTIVATION backward
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    dZ = sigmoid_backward(dA, activation_cache) if activation == "sigmoid" else relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# [LINEAR -> RELU] * (L - 1) -> LINEAR -> SIGMOID backward
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    # print(Y.shape)
    # print(AL.shape)
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l+1)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['dA' + str(l+2)],current_cache,'relu')

    return grads

# AL, Y_assess, caches = testCases.L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print(grads)

# 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

    return parameters

# parameters, grads = testCases.update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
# print(parameters)

# 导入数据
train_x_orig, train_y, test_x_orig, test_y, classes = lr_utils.load_dataset()

# plt.imshow(train_x_orig[10])
# plt.show()
# print(train_y[0,7])
# print(train_x_orig.shape)
# print(train_y.shape)

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1)
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)

# 标准化
train_x = train_x_flatten.T / 255.0
test_x = test_x_flatten.T / 255.0

# print(train_x.shape)
# print(test_x.shape)

# L层神经网络
layers_dims = [12288, 20, 7, 5, 1]

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)

    costs = []
    parameters = init_deep(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

# def predict(X, Y, parameters):
#     AL, caches = L_model_forward(X, parameters)
#     prediction = np.round(AL)
#     return 100 - np.mean(np.abs(prediction - Y))
# pred_train = predict(train_x, train_y, parameters)
# pred_test = predict(test_x, test_y, parameters)
# print(pred_train)
# print(pred_test)

# 载入图片
# img = Image.open("./datasets/images.webp")
# resize_img = img.resize((64, 64))
# np_img = np.array(resize_img)
# test_img = np.resize(np_img, (1, 64, 64, 3))
# test_img = test_img.reshape(-1,1) / 255.0
# print(test_img.shape)

# 预测
# AL, caches = L_model_forward(test_img, parameters)
# prediction = np.round(AL)
# print(prediction)
