import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# plt.show()

# 马尔可夫假设
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# print(features.shape)
# print(labels.shape)
# print(x.shape)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print('epoch %d, loss %f' % (epoch + 1, d2l.evaluate_loss(net, train_iter, loss)))

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测
onestep_preds = net(features)
# d2l.plot(
#     [time, time[tau:]],
#     [x.detach().numpy(), onestep_preds.detach().numpy()],
#     'time', 'x',
#     legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3)
# )
# plt.show()

# 使用预测的数据继续预测
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape(1, -1))

# d2l.plot([time, time[tau:], time[n_train + tau:]],
#     [x.detach().numpy(),
#     onestep_preds.detach().numpy(),
#     multistep_preds[n_train + tau:].detach().numpy()],
#     'time', 'x',
#     legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3)
# )
# plt.show()

max_step = 64

features = torch.zeros((T - tau - max_step + 1, tau + max_step))
for i in range(tau):
    features[:, i] = x[i:T - tau + i - max_step + 1]

for i in range(tau, tau + max_step):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i -1:T - max_step + i] for i in steps],
         [features[:, (tau + i -1)].detach().numpy() for i in steps],
         'time', 'x',
         legend=[f'{i}-step' for i in steps],
         xlim=[5, 1000], figsize=(6, 3))
plt.show()