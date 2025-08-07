import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time  # 引入time模块，可以在循环中加入短暂延时，方便观察

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --- 步骤 1: 数据和函数准备 (与之前相同) ---
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([4, 6, 8, 10, 12])


# 使用我们修正好的、参数顺序正确的 model 函数
def model(x, w, b):
    return w * x + b


# 使用我们最开始的、基于循环的 cost_function，便于理解
def cost_function(w, b, X, Y):
    total_cost = 0
    for i in range(len(X)):
        y_predicted = model(X[i], w, b)
        total_cost += (y_predicted - Y[i]) ** 2
    return total_cost / len(X)


# --- 步骤 2: 初始化变量和范围 ---
# 为了让控制台输出不至于刷屏太快，我们使用较小的范围和步长
w_range = np.arange(0.0, 4.1, 0.5)
b_range = np.arange(0.0, 4.1, 0.5)

# 初始化一个列表，用于存储所有计算出的cost值，为最终的3D绘图做准备
costs_grid = []
# 初始化追踪最优解的变量
min_cost = float('inf')
best_w = 0
best_b = 0

print("--- 开始在参数空间中搜索最优解 ---")
print(f"w 的搜索范围: {w_range}")
print(f"b 的搜索范围: {b_range}")
print("-" * 35)

# --- 步骤 3: 使用嵌套循环计算，并实时打印过程 ---
# 外层循环遍历 b
for b in b_range:
    # 临时列表，存放当前行（固定b，变化w）的cost
    row_costs = []
    # 内层循环遍历 w
    for w in w_range:
        # 计算当前 (w, b) 组合的 Cost
        cost = cost_function(w, b, x_data, y_data)
        row_costs.append(cost)

        # --- 在控制台输出当前步骤 ---
        print(f"尝试: w={w:.1f}, b={b:.1f}  =>  计算出的 Cost = {cost:.4f}")

        # 检查是否找到了一个新的最小 Cost
        if cost < min_cost:
            old_min_cost = min_cost
            min_cost = cost
            best_w = w
            best_b = b
            # 如果找到新低点，高亮显示
            print(
                f"  ==> 发现新低点! Cost从 {old_min_cost:.4f} 降至 {min_cost:.4f}。最佳参数更新为 (w={best_w:.1f}, b={best_b:.1f})")

        # time.sleep(0.01) # 如果觉得输出太快，可以取消这行注释，加入0.01秒延时

    # 将一整行w对应的costs存入我们的网格中
    costs_grid.append(row_costs)

print("-" * 35)
print("--- 搜索完成 ---")
print(f"在所有尝试中，找到的全局最优解为:")
print(f"w = {best_w:.1f}, b = {best_b:.1f}, 对应的最小 Cost = {min_cost:.4f}")

# --- 步骤 4: 绘制三维表面图 (与之前类似) ---
print("\n--- 正在生成三维可视化图表 ---")

# 将 list of lists 转换为 numpy array，以便绘图
costs_grid = np.array(costs_grid)

# 使用 np.meshgrid 创建绘图所需的坐标网格
W, B = np.meshgrid(w_range, b_range)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 注意：这里 Z轴 用的数据是 costs_grid
# costs_grid 的维度可能和 W, B 不匹配，需要转置 (Transpose)
# 因为我们的外层循环是b，内层是w，所以costs_grid的维度是(len(b), len(w))
# 而meshgrid生成的W,B的维度是(len(b), len(w))，正好匹配，无需转置。
ax.plot_surface(W, B, costs_grid, cmap='viridis', edgecolor='none')

ax.set_xlabel('权重 (w)')
ax.set_ylabel('偏置 (b)')
ax.set_zlabel('代价函数 (Cost / MSE)')
ax.set_title('线性模型的代价函数三维视图')

# 标记出最低点
ax.scatter(best_w, best_b, min_cost, color='red', s=100, label=f'最低点 (w={best_w}, b={best_b})')
ax.legend()

plt.show()