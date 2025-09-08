# 初始化、正则化、梯度检验
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils
import reg_utils
import gc_utils

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 初始化
# train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=False)
# plt.show()

# 用于初始化练习的3层神经网络模型
def model_init(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization='he'):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # 初始化参数
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):
        a3, cache = init_utils.forward_propagation(X, parameters)
        cost = init_utils.compute_loss(a3, Y)
        grads = init_utils.backward_propagation(X, Y, cache)
        parameters = init_utils.update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# 零初始化
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_zeros([3,2,1])
# print(parameters)
# print(parameters['b2'].shape)

# parameters = model_init(train_X, train_Y, initialization='zeros')
# print('On the train set:')
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print('On the test set:')
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
# print('predictions_train = ' + str(predictions_train))
# print('predictions_test = ' + str(predictions_test))

# # 绘制决策边界
# plt.title('Model with Zeros initialization')
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

# 随机初始化
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_random([3, 2, 1])
# print(parameters)

# parameters = model_init(train_X, train_Y, initialization='random')
# print('On the train set:')
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print('On the test set:')
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
# print('predictions_train = ' + str(predictions_train))
# print('predictions_test = ' + str(predictions_test))

# # 绘制决策边界
# plt.title('Model with large random initialization')
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

# He初始化
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2.0 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_he([2, 4, 1])
# print(parameters)

# parameters = model_init(train_X, train_Y, initialization='he')
# print('On the train set:')
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print('On the test set:')
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
# print('predictions_train = ' + str(predictions_train))
# print('predictions_test = ' + str(predictions_test))

# # 绘制决策边界
# plt.title('Model with He initialization')
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

# 正则化
train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)
# plt.show()

# 用于正则化练习的模型
def model_reg(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = reg_utils.initialize_parameters(layers_dims)

    for i in range(num_iterations):
        if keep_prob == 1:
            a3, cache = reg_utils.forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        if lambd == 0:
            cost = reg_utils.compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        assert (lambd == 0 or keep_prob == 1)

        if lambd == 0 and keep_prob == 1:
            grads = reg_utils.backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_dropout(X, Y, cache, keep_prob)

        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# parameters = model_reg(train_X, train_Y)
# print('On the training set:')
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print('On the test set:')
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title('Model without regularization')
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# L2正则化
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    corss_entropy_cost = reg_utils.compute_cost(A3, Y)
    L2_regularization_cost = (1. / m * lambd / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = corss_entropy_cost + L2_regularization_cost

    return cost

# L2正则化的反向传播
def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1. / m * np.dot(dZ3, A2.T) + lambd / m * W3
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {'dZ3': dZ3, 'dW3': dW3, 'db3': db3, 'dA2': dA2,
                 'dZ2': dZ2, 'dW2': dW2, 'db2': db2, 'dA1': dA1,
                 'dZ1': dZ1, 'dW1': dW1, 'db1': db1}

    return gradients

# parameters = model_reg(train_X, train_Y, lambd=0.7)
# print('On the training set:')
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print('On the test set:')
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title('Model with L2-regularization')
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# dropout
def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    np.random.seed(1)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = reg_utils.relu(Z1)

    # 设置掩码（失活矩阵）
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 /= keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 /= keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1,
             Z2, D2, A2, W2, b2,
             Z3, A3, W3, b3)

    return A3, cache

# dropout的反向传播
def backward_propagation_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2
    dA2 /= keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 /= keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {'dZ3': dZ3, 'dW3': dW3, 'db3': db3, 'dA2': dA2,
                 'dZ2': dZ2, 'dW2': dW2, 'db2': db2, 'dA1': dA1,
                 'dZ1': dZ1, 'dW1': dW1, 'db1': db1}

    return gradients

# parameters = model_reg(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)
# print('On the training set:')
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print('On the test set:')
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title('Model with dropout')
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# 梯度检验

# 一维梯度检查    J(θ) = θ * x
def forward_propagation(x, theta):
    J = theta * x
    return J

def backward_propagation(x, theta):
    dtheta = x
    return dtheta

def gradient_check(x, theta, epsilon=1e-7):
    gradapprox = (forward_propagation(x, theta + epsilon) - forward_propagation(x, theta - epsilon)) / (2 * epsilon)
    grad = backward_propagation(x, theta)
    difference = np.linalg.norm(grad - gradapprox) / (np.linalg.norm(grad) + np.linalg.norm(gradapprox))

    if difference < 1e-7:
        print('The gradient is correct')
    else:
        print('The gradient is wrong')

    return difference

x, theta = 2, 4
difference = gradient_check(x, theta)
print(difference)

# N维梯度检验 LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
def forward_propagation_n(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = gc_utils.relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = gc_utils.relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = gc_utils.sigmoid(Z3)

    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {'dZ3': dZ3, 'dW3': dW3, 'db3': db3,
                 'dA2': dA2, 'dZ2': dZ2, 'dW2': dW2, 'db2': db2,
                 'dA1': dA1, 'dZ1': dZ1, 'dW1': dW1, 'db1': db1}

    return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    grad = gc_utils.gradients_to_vector(gradients)
    parameters_values = gc_utils.dictionary_to_vector(parameters)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapporx = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        J_plus, _ = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] -= epsilon
        J_minus, _ = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(theta_minus))

        gradapporx[i] = (J_plus - J_minus) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapporx)
    denominator = np.linalg.norm(gradapporx) + np.linalg.norm(grad)
    difference = numerator / denominator

    if difference < 1e-7:
        print('The gradient is correct, difference = ' + str(difference))
    else:
        print('The gradient is wrong, difference = ' + str(difference))

    return difference

# 缺少梯度检验的数据，无法实际运行
# X, Y, parameters = gradient_check_n_test_case()
#
# cost, cache = forward_propagation_n(X, Y, parameters)
# gradients = backward_propagation_n(X, Y, cache)
# difference = gradient_check_n(parameters, gradients, X, Y)
