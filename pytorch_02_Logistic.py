import torch
import numpy as np
import matplotlib.pyplot as plt
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#逻辑回归，二分类问题
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0], [1.0]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入层到输出层的线性变换

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # 使用sigmoid函数将输出映射到0-1之间

# 实例化模型
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction='mean')  # 二分类交叉熵损失函数

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义优化器

# 训练模型
for epoch in range(1000):

    y_pred = model(x_data)  # 前向传播，计算预测值
    loss = criterion(y_pred, y_data)  # 计算损失
    print(epoch, '损失:', loss.item())  # 打印每个epoch的损失

    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新参数

# 训练模型
x_test = torch.Tensor([[4.0]])  # 测试数据
with torch.no_grad():
    y_test = model(x_test)  # 预测

print(f'\n输入为 4.0 的预测概率为: {y_test.item():.4f}')
