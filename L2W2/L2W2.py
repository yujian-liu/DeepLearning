# 优化算法
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from scipy.linalg import cossin

import opt_utils
import testCase

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 梯度下降 GD
def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

    return parameters

# parameters, grads, learning_rate = testCase.update_parameters_with_gd_test_case()
#
# parameters = update_parameters_with_gd(parameters, grads, learning_rate)
# print(parameters)

# 随机梯度下降 SGD (伪代码)
# X = data_input
# Y = labels
# parameters = init_parameters(layers_dims)
# for i in range(num_iterations):
#     for j in range(m):
#         a, cache = forward_propagation(X[:, j], parameters)
#         cost = compute_cost(a, Y[:, j])
#         grads = backward_propagation(a, cache, parameters)
#         parameters = update_parameters(parameters, grads)

# Mini-batch
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 打乱
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # 分组
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 无法整除时处理剩余
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# X_assess, Y_assess, mini_batch_size = testCase.random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
# print(X_assess.shape)
# print(mini_batches[0][0][0][0:3])

# 动量梯度下降
# 初始化速度
def init_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return v

# 带动量的参数更新
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        v['dW' + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        parameters['W' + str(l + 1)] -= learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * v['db' + str(l + 1)]

    return parameters, v

# parameters, grads, v = testCase.update_parameters_with_momentum_test_case()
# parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
# print(parameters)
# print(v)

# Adam
def init_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
        s['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        s['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v, s

# 参数更新
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - beta1 ** t)
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - beta1 ** t)

        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * (grads['dW' + str(l + 1)] ** 2)
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * (grads['db' + str(l + 1)] ** 2)

        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta2 ** t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta2 ** t)

        parameters['W' + str(l + 1)] -= learning_rate * (v_corrected['dW' + str(l + 1)] / np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon))
        parameters['b' + str(l + 1)] -= learning_rate * (v_corrected['db' + str(l + 1)] / np.sqrt(s_corrected['db' + str(l + 1)] + epsilon))

    return parameters, v, s

# parameters, grads, v, s = testCase.update_parameters_with_adam_test_case()
# parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
# print(parameters)
# print(v)
# print(s)

# 不同优化算法的模型
train_X, train_Y = opt_utils.load_dataset(is_plot=False)
# plt.show()

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    parameters = opt_utils.initialize_parameters(layers_dims)

    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        v = init_velocity(parameters)
    elif optimizer == 'adam':
        v, s = init_adam(parameters)

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = opt_utils.forward_propagation(minibatch_X, parameters)
            cost = opt_utils.compute_cost(a3, minibatch_Y)
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, caches)

            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == 'adam':
                t += 1  # 使用adam的次数
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
                                                               beta1, beta2, epsilon)
        if print_cost and i % 1000 == 0:
            print('Cost after iteration %i: %f' % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# 小批量梯度下降
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer='gd')
#
# predictions = opt_utils.predict(train_X, train_Y, parameters)
#
# plt.title('Model with Gradient Descent optimization')
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
# plt.show()

# 带动量的小批量梯度下降
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer='momentum')
#
# predictions = opt_utils.predict(train_X, train_Y, parameters)
#
# plt.title('Model with Momentum optimization')
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
# plt.show()

# Adam模式的小批量梯度下降
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer='adam')
#
# predictions = opt_utils.predict(train_X, train_Y, parameters)
#
# plt.title('Model with Adam optimization')
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
# plt.show()