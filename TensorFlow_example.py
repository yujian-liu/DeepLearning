# import numpy as np
# import tensorflow as tf
#
# w = tf.Variable(0, dtype=tf.float32)
# x = tf.placeholder(tf.float32, [3, 1])
# # cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)
# # cost = w**2 - 10 * w + 25
# cost = x[0][0] * w**2 + x[1][0] * w + x[2][0]
# train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
# init = tf.global_variables_initializer()
# session = tf.Session()
# session.run(init)
# print(session.run(w))
#
# session.run(train)
# print(session.run(w))
#
# for i in range(1000):
#     session.run(train)
# print(session.run(w))

# 上述代码课程中的代码，是TensorFlow 1.x版本，无法运行，以下为通义千问给出的2.x版本
import tensorflow as tf
import numpy as np

# 定义变量 w，初始值为 0.0
w = tf.Variable(0.0, dtype=tf.float32)

# 定义输入 x（模拟 placeholder 的作用）
x = tf.constant([[1.0], [-10.0], [25.0]])  # 对应 w² -10w +25

# 定义损失函数（相当于 cost = x[0][0]*w² + x[1][0]*w + x[2][0]）
def compute_loss():
    return x[0][0] * w**2 + x[1][0] * w + x[2][0]

# 使用 TF 2.x 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 打印初始 w
print("Initial w:", w.numpy())

# 执行一次训练
with tf.GradientTape() as tape:
    loss = compute_loss()
grads = tape.gradient(loss, [w])
optimizer.apply_gradients(zip(grads, [w]))
print("After 1 step:", w.numpy())

# 再训练 1000 步
for i in range(1000):
    with tf.GradientTape() as tape:
        loss = compute_loss()
    grads = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(grads, [w]))

print("After 1000 steps:", w.numpy())