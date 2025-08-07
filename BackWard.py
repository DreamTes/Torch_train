import torch

# 反向传播示例：使用 PyTorch 实现简单的线性回归
# 准备数据集
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

w = torch.tensor([0.0], requires_grad=True)  # 初始化权重 requires_grad=True 表示需要计算梯度

# 学习率
learning_rate = 0.01

def model(x):
    return w * x

def loss_function(y_pred, y_true):
    return (y_pred - y_true) ** 2  # 均方误差损失函数

for epoch in range(10):
    for i in range(len(x_data)):
        # 1. 前向传播：计算预测值
        y_pred = model(x_data[i])

        # 2. 计算损失
        loss = loss_function(y_pred, y_data[i])

        # 3. 反向传播：计算梯度
        loss.backward()

        # 4. 更新权重
        with torch.no_grad():
            w-= learning_rate * w.grad

        # 5. 清零梯度
        w.grad.data.zero_()
    # 在每个epoch结束后打印一次信息
    # 我们需要重新计算一下整个数据集上的loss来观察整体效果
    with torch.no_grad():
        total_loss = torch.mean((model(x_data) - y_data) ** 2)
        print(f'Epoch {epoch + 1}: w = {w.item():.4f},  Total Loss = {total_loss.item():.4f}')

# 打印最终的权重
print(f'最终的权重 w = {w.item():.4f}')

