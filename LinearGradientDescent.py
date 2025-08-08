import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.timeTools import epoch_diff

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建训练数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# 建立线性模型
def model(x, w):
    return w * x

# 定义代价函数 (Cost Function - MSE)
def cost_function(w, X, Y):
    total_cost = 0
    num_samples = len(X)
    for i in range(num_samples):
        # 1. 对单个样本进行预测
        y_predicted = model(X[i], w)
        # 2. 计算这个样本的 loss
        y_true = Y[i]
        single_loss = (y_predicted - y_true) ** 2
        # 3. 累加所有样本的 loss
        total_cost += single_loss
    return total_cost / num_samples

def gradient(w, X, Y):
    grad = 0 # 初始化梯度
    num_samples = len(X)
    for i in range(num_samples):
        y_predicted = model(X[i], w)
        grad += 2 * (y_predicted - Y[i]) * X[i]
    return grad / num_samples

w_list = [] # 权重列表
mse_list = [] # 均方误差列表
epochs = [] # 训练次数列表

# 学习率 (Learning Rate): 每一步迈多大
learning_rate = 0.01
# 初始的w，可以随便设一个值
w = 0.0
# 训练次数 (Epochs): 一共走多少步

for epoch in range(30):
    # 计算当前的均方误差
    mse = cost_function(w, x, y)
    w_list.append(w)
    mse_list.append(mse)
    epochs.append(epoch + 1)

    # 打印当前的w和均方误差
    print(f'第 {epoch + 1} 步: 当 w={w:.2f} 时，均方误差 (MSE) = {mse:.2f}')

    # 计算梯度
    grad = gradient(w, x, y)

    # 更新权重
    w -= learning_rate * grad

# 绘制权重和均方误差的折线图
plt.plot(epochs, mse_list)
plt.title('梯度下降法训练过程中的均方误差变化')
plt.xlabel('训练次数 (Epochs)')
plt.ylabel('均方误差 (MSE)')
plt.grid()
plt.xticks(epochs)  # 确保x轴显示每个epoch
plt.show()








