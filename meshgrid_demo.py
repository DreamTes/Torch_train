import numpy as np
import matplotlib.pyplot as plt
# 1. 创建坐标向量
x_coords = np.linspace(-5, 5, 100)
y_coords = np.linspace(-5, 5, 100)

# 2. 创建网格
X, Y = np.meshgrid(x_coords, y_coords)

# 3. 计算函数 Z = sin(sqrt(X^2 + Y^2))
Z = np.sin(np.sqrt(X**2 + Y**2))

# 4. 使用 matplotlib 绘制等高线图
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()
plt.title("Z = sin(sqrt(X^2 + Y^2))")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()