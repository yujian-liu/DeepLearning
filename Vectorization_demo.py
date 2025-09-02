import numpy as np
import time

a = np.array([1,2,3,4])
print(a)

a = np.random.rand(1000000)
b = np.random.rand(1000000)

# 向量化
tic = time.time()
c1 = np.dot(a,b)
toc = time.time()

print(c1)
print("Vectorized version:" + str(1000 * (toc - tic)) + "ms")

# 非向量化
c2 = 0
tic = time.time()
for i in range(1000000):
    c2 += a[i] * b[i]
toc = time.time()

print(c2)
print("For loop:" + str(1000 * (toc - tic)) + "ms")

# 发现向量化与非向量化的结果不一致，进行进一步判断
# 应该是由于浮点数计算导致的误差
if np.allclose(c1, c2, atol=1e-10):
    print("结果在精度范围内一致 ✅")
else:
    print("结果差异较大 ❌")