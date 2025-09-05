# 用神经网络思想实现Logistic回归
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# 数据加载
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 测试数据
# print(train_set_y.shape)
# index = 5
# plt.imshow(train_set_x_orig[index])
# print("y = " + str(train_set_y[:, index]) + " it is " + classes[np.squeeze(train_set_y[:, index])].decode("utf-8"))

# reshape
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 初始化参数
def init(dim):
    w = np.zeros((dim,1))
    b = 0
    assert (w.shape == (dim,1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

# 前向传播和后向传播
def propagate(w, b, X, Y):
    m = X.shape[1]

    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 后向传播
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return cost, grads

# 优化函数
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        cost, grads = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录数据
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

# 预测函数
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    assert (Y_prediction.shape == (1,m))

    return Y_prediction

# 整合为模型
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = init(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w, b = parameters["w"], parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,}

    return d

num_iterations = 2000
learning_rate = 0.005
print_cost = True
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, print_cost)

# 绘制损失函数和梯度
costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# 不同学习率
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate = " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, i, print_cost)
    print('\n' + "----------------------------")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()