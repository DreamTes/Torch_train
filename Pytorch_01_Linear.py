import torch
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 反向传播示例：使用 PyTorch 实现简单的线性回归
# 准备数据集
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # 调整输入形状为二维张量
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])  # 调整目标形状为二维张量

# 建立线性回归模型
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__() # 初始化父类
        self.linear = torch.nn.Linear(1, 1)  # 定义线性层

    def forward(self, x):
        return self.linear(x)
# 实例化模型
model = LinearRegressionModel() # 实例化模型

criterion = torch.nn.MSELoss(reduction='mean')  # 定义损失函数

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义优化器

# 使用批量梯度下降法（Batch Gradient Descent）进行训练

# 批量梯度下降法（Batch Gradient Descent）进行训练
for epoch in range(100):
    # 1. 前向传播：计算预测值
    y_pred = model(x_data)  # 使用整个数据集进行前向传播

    # 2. 计算损失
    loss = criterion(y_pred, y_data)  # 使用整个数据集计算损失

    print('Epoch [{}/100], Loss: {:.4f}'.format(epoch + 1, loss.item()))  # 打印每个epoch的损失

    # 3. 反向传播：计算梯度
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播

    # 4. 更新权重
    optimizer.step()  # 更新参数


print('w =',model.linear.weight.item())# 打印最终的权重
print('b =',model.linear.bias.item())  # 打印偏置

x_test = torch.Tensor([[6.0]])  # 测试数据
y_test = model(x_test)  # 预测
print('预测值:', y_test.item())  # 打印预测结果

# 绘制结果
plt.scatter(x_data.numpy(), y_data.numpy(), label='真实数据', color='blue')  # 绘制真实数据点
plt.plot(x_data.numpy(), model(x_data).detach().numpy(), label='拟合直线', color='red')  # 绘制拟合直线
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合')
plt.legend()
plt.show()  # 显示图形