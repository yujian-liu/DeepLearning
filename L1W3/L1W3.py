# 用1层隐藏层的神经网络分类二维数据
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) #设置一个固定的随机种子，以保证接下来的步骤中结果是一致的。

# 数据加载
X, Y = load_planar_dataset()

# print(X.shape)
# print(Y.shape)

# 数据可视化
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral)
# plt.show()

# 逻辑回归
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())

# 绘制决策边界
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title('Logistic Regression')
plt.show()

LR_predictions = clf.predict(X.T)
accuracy = (np.dot(Y,LR_predictions) + np.dot(1-Y,LR_predictions)) / float(Y.size) * 100
print("Accuracy of Logistic Regression is: %d " % accuracy.item())

# 神经网络
def layer_sizes(X, Y):
    n_x = X.shape[0]    #输入大小
    n_h = 4             #隐藏层大小
    n_y = Y.shape[0]    #输出大小
    return (n_x, n_h, n_y)

# 初始化参数
def init(n_x, n_h, n_y):
    np.random.seed(2)

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

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = init(n_x, n_h, n_y)
# print(parameters)

# 正向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))

# 损失函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]

    logprods = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = -np.sum(logprods) / m

    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost

# 反向传播
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    A1 = cache["A1"]
    A2 = cache["A2"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

# 更新参数
def update_parameters(grads, parameters, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# 构建模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x, n_y = layer_sizes(X, Y)[0], layer_sizes(X, Y)[2]
    parameters = init(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(grads, parameters)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

# X_assess, Y_assess = nn_model_test_case()
#
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# 预测函数
def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    prediction = np.round(A2)

    return prediction

parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)

plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)
plt.title('Decision Boundary for hidden layer size ' + str(4))
plt.show()

predictions = predict(X, parameters)
accuracy = (np.dot(Y,predictions.T) + np.dot(1 - Y,1 - predictions.T)) / float(Y.size) * 100
print(accuracy)
print('Accuracy of hidden layer size predictions: %d' % accuracy.item())

# 不同隐藏层大小
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)
    predictions = predict(X, parameters)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()