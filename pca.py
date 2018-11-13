import numpy as np
import matplotlib.pylab as plt
import random
from sklearn import datasets
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

def pca(X, m, k):  # X：数据矩阵n*m,n是维度,m是数据个数.k是降到的维度
    # 均值化
    mean = np.mean(X, axis=1)  # axis=1,算出每一行的均值
    mean = np.tile(mean, (1, m))  # tile,按样子复制1*m个样子
    X = X - mean
    # 求对称矩阵：协方差矩阵;求C矩阵的特征向量# Cov = np.cov(X)
    C = 1.0 / m * X * X.T

    eig_value, eig_vetor = np.linalg.eig(C)  # 按列排放着特征值和对应的特征向量,得到的特征向量已经是单位向量了
    # print("eig_value\n", eig_value, "\neig_vetor\n", eig_vetor)
    eigsort_value = np.argsort(-eig_value)  # 降序排列，argsort得到的是排序序号
    eigsort_vetor = eig_vetor[:, eigsort_value]  # 列按照上述排列方式排列
    # print("eig_sort\n", eigsort_value, "\nredEig\n", eigsort_vetor)

    # 降维
    P = eigsort_vetor.T[:k, :]
    Y = P * X
    # print("Y\n", Y)

    # 重构
    rev = P.T * Y + mean
    # print("rev\n", rev)

    # 误差函数
    loss_vector = X - rev
    loss = np.sum(np.multiply(loss_vector, loss_vector))
    # print("loss\n",loss)

    return Y


# ###############################
# # 测试两维降到一维，取观测值x.y,y=x基础上加高斯噪声
# a = []
# b = []
# x = 0
# xnum = 100  # 数据集x数目，可以更改
#
#
# def func(x):
#     mu = 0  # 均值
#     sigma = 0.15  # 方差
#     epsilon = random.gauss(mu, sigma)  # 高斯分布随机数
#     return epsilon
#
#
# for i in range(0, xnum):
#     x = x + 1.0 / xnum
#     a.append(x)
# for i in range(0, xnum):
#     b.append(a[i] + func(a[i]))
# m = xnum
# D = 2
# X = np.mat(np.ones((D, m)))
# X[0, :] = a
# X[1, :] = b
# print("preX\n", X)
# pca(X, m, 1)
##################################

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target
new_X = pca(X,206669376,3)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.nipy_spectral)
plt.show()



# plt.figure(1, dpi=100)
# plt.figure()
# plt.xlabel("X axis")
# plt.ylabel("Y axis")
# plt.scatter(a, b, c='c', alpha=0.4)
# x = np.linspace(0, 1, 100)  # 中间间隔100个元素
# plt.plot(x, x, color="r", label='sin(2$\pi$x)')
# # 显示所画的图
# plt.show()

# X = np.mat(np.arange(2, 6, 1).reshape(2, 2))
# Y = np.mat(np.arange(4).reshape(2, 2))
# print(X, "\n", Y)
# print(np.sum(X))
