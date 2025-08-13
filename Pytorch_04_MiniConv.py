import torch
import torch .nn as nn

### 定义参数
learning_rate = 0.1

# 输入图片 X (4x4)
# PyTorch的卷积层要求输入是 [Batch, Channel, Height, Width] 的4D张量
X = torch.tensor([[
    [1., 1., 1., 0.],
    [0., 1., 1., 1.],
    [0., 0., 1., 1.],
    [0., 0., 0., 1.]
]], dtype=torch.float32).reshape(1, 1, 4, 4) # 1个样本，1个通道，4x4的图像

y_true = torch.tensor([1.], dtype=torch.float32) # 真实标签

# 定义卷积核和偏置量
W_conv_init = torch.tensor([[
    [1., 0., 1.],
    [0., 1., 0.],
    [1., 0., 1.]
]], dtype=torch.float32).reshape(1, 1, 3, 3) # [out_channels, in_channels, H, W]
b_conv_init = torch.tensor([1.], dtype=torch.float32)

# 全连接层权重 W_fc (1x4) 和偏置 b_fc
W_fc_init = torch.tensor([[1., -1., 0., 1.]], dtype=torch.float32) # [out_features, in_features]
print("初始化的 W_fc:\n", W_fc_init)
b_fc_init = torch.tensor([-1.], dtype=torch.float32)
print("初始化的 b_fc:\n", b_fc_init)

### 定义卷积层

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)  # 输入通道=1, 输出通道=1, 卷积核大小=3x3, 步幅=1, 填充=0
relu_layer = nn.ReLU() #  ReLU激活函数

fc_layer = nn.Linear(4, 1) # 全连接层，输入特征数=4, 输出特征数=1

# 用设定的初始值来覆盖PyTorch的随机初始值
with torch.no_grad():  # 在不需要梯度计算的上下文中初始化权重和偏置
    conv_layer.weight.data = W_conv_init  # 设置卷积层的权重
    conv_layer.bias.data = b_conv_init  # 设置卷积层的偏置
    fc_layer.weight.data = W_fc_init  # 设置全连接层的权重
    fc_layer.bias.data = b_fc_init  # 设置全连接层的偏置

# forward pass
Z_conv = conv_layer(X)  # 卷积计算
print("卷积层输出 Z_conv:\n", Z_conv.data)
Z_relu = relu_layer(Z_conv)  # ReLU激活函数输出
print("ReLU激活函数输出 Z_relu:\n", Z_relu.data)
Z_flat = Z_relu.reshape(-1, 4)  # 展平操作，将输出从 [Batch, Channel, Height, Width] 转换为 [Batch, Features], -1表示自动计算特征数
y_pred = fc_layer(Z_flat)  # 全连接层输出
loss = (y_pred - y_true).pow(2).mean()  # 均方误差损失函数
# backward pass
loss.backward()  # 计算梯度
# 更新参数
with torch.no_grad():  # 在不需要梯度计算的上下文中更新参数
    conv_layer.weight.data -= learning_rate * conv_layer.weight.grad  # 更新卷积层权重
    conv_layer.bias.data -= learning_rate * conv_layer.bias.grad  # 更新卷积层偏置
    fc_layer.weight.data -= learning_rate * fc_layer.weight.grad  # 更新全连接层权重
    fc_layer.bias.data -= learning_rate * fc_layer.bias.grad  # 更新全连接层偏置

print("更新后的 W_fc:\n", fc_layer.weight.data)
print("更新后的 b_fc:\n", fc_layer.bias.data)
