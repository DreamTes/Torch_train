import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D 绘图工具

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 训练数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 6, 8, 10, 12])

# 建立线性模型
def model(x, w, b):
    return w * x + b

# np.linspace(start, stop, num) 会生成一个从 start 到 stop 的等差数列，包含 num 个点
w_range = np.linspace(0.0, 4.0, 101)  # 权重范围
b_range = np.linspace(0.0, 4.0, 101)  # 偏置范围
# 使用 np.meshgrid 创建二维的坐标网格
# W 和 B 都是 101x101 的二维数组
W, B = np.meshgrid(w_range, b_range)
# 初始化一个和 W、B 形状完全一样的全零数组，用于存放每个(w, b)点对应的Cost值
Cost = np.zeros(W.shape)

# 遍历每一个数据点 (x_data[i], y_data[i])
for i in range(len(x)):
    # 对于每一个数据点，计算整个(W, B)网格的预测值与真实值的差的平方
    # 这一步是向量化计算的核心：
    # W, B 是 101x101 的网格，x_data[i] 是一个标量。
    # Numpy的广播机制会自动处理，一次性计算出所有101*101个点的预测值。
    y_pred_grid = model(x[i], W, B)

    # 计算误差的平方，并累加到Cost网格中
    Cost += (y_pred_grid - y[i]) ** 2
# 最后，除以数据点的数量，得到所有点的平均误差(MSE)
Cost = Cost / len(x)

# --- 步骤 4: 绘制三维表面图 ---
# 1. 创建一个图形实例
fig = plt.figure(figsize=(10, 7))

# 2. 在图形中添加一个三维坐标系
ax = fig.add_subplot(111, projection='3d')

# 3. 绘制三维表面
# W, B 是 x, y 坐标网格，Cost 是 z 坐标(高度)
# cmap='viridis' 是一种流行的颜色映射方案，让高度不同的地方颜色不同
surface = ax.plot_surface(W, B, Cost, cmap='viridis')

# 4. 添加坐标轴标签和标题
ax.set_xlabel('权重 (w)', fontsize=12)
ax.set_ylabel('偏置 (b)', fontsize=12)
ax.set_zlabel('代价函数 (Cost / MSE)', fontsize=12)
ax.set_title('线性模型的代价函数三维视图', fontsize=15)

# 5. 添加颜色条，用于说明颜色与Cost值的对应关系
fig.colorbar(surface, shrink=0.5, aspect=5)

# 6. 显示图形
plt.show()








