import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 线性模型的实现 y = 2x+2
x = np.array([1, 2, 3, 4, 5])  # 房屋面积
y = np.array([2, 4, 6, 8, 10])  # 房屋价格

# 建立线性模型
def model(x, w):
    return w * x

# 定义代价函数 (Cost Function - MSE)
def cost_function(w,X, Y):
    total_cost = 0
    num_samples = len(X)
    for i in range(num_samples):
        # 1. 对单个样本进行预测
        y_predicted = model(X[i], w,)
        # 2. 计算这个样本的 loss
        y_true = Y[i]
        single_loss = (y_predicted - y_true) ** 2
        # 3. 累加所有样本的 loss
        total_cost += single_loss
    # 4. 计算平均值，得到最终的 Cost
    return total_cost / num_samples

w_list = [] # 权重列表
mse_list = [] # 均方误差列表

for w in np.arange(0.0, 4.1, 0.1): # 调整w范围以便观察
    mse = cost_function(w, x, y)
    w_list.append(w)
    mse_list.append(mse)
    print(f'当 w={w:.1f} 时，均方误差 (MSE) = {mse:.2f}')

# 绘制权重和均方误差的折线图
plt.plot(w_list, mse_list)
plt.title('权重与均方误差的关系')
plt.xlabel('权重 (w)')
plt.ylabel('均方误差 (MSE)')
plt.grid(True)
plt.show()









