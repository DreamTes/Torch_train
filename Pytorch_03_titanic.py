import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# pytorch实现预测顾客是否生存的模型

raw_data = pd.read_csv("dataset/train.csv")
# print(raw_data.head()) # 打印前5行数据
# print("数据集的列名:", raw_data.columns) # 打印数据集的列名
# print("打印数据集的信息:",raw_data.info) # 打印数据集的信息
# print("数据集的形状:", raw_data.shape) # 打印数据集的形状
# print("数据集的描述统计信息:\n", raw_data.describe()) # 打印数据集的描述统计信息
# print(raw_data.isnull().sum())  # 打印每列的缺失值数量
# 数据预处理，舍弃不需要的特征
raw_data = raw_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # 删除不需要的列
# 填充缺失值
raw_data['Age'] = raw_data['Age'].fillna(raw_data['Age'].mean())  # 用年龄的均值填充缺失值
raw_data['Pclass'] = raw_data['Pclass'].fillna(raw_data['Pclass'].mode()[0])  # 用最常见的等级填充缺失值
# 将部分特征进行One-Hot Encoding 处理
raw_data = pd.get_dummies(raw_data, columns=['Sex', 'Embarked'], drop_first=True, dtype=float)  # 性别和登船港口进行独热编码
# 分离特征和标签
X = raw_data.drop('Survived', axis=1)
y = raw_data['Survived']

# 使用 train_test_split 进行划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 划分训练集和验证集，保持标签分布一致

# 特征缩放: 在训练集上fit，然后在训练集和验证集上transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val) # 注意：验证集只用transform

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.to_numpy().astype(np.float32)).view(-1, 1)
X_val_tensor = torch.from_numpy(X_val_scaled.astype(np.float32))
y_val_tensor = torch.from_numpy(y_val.to_numpy().astype(np.float32)).view(-1, 1)

# 数据集准备与划分，建立dataset和dataloader
class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features) # 返回数据集的长度

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# 划分训练集和验证集
train_data = TitanicDataset(X_train_tensor, y_train_tensor)  # 创建训练集
val_data = TitanicDataset(X_val_tensor, y_val_tensor)  # 创建验证集
# 创建DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 模型设计
class TitanicModel(torch.nn.Module):
    def __init__(self,num_features):
        super(TitanicModel, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 16)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(16, 8)   # 隐藏层到隐藏层
        self.fc3 = torch.nn.Linear(8, 1)    # 隐藏层到输出层
        self.relu = torch.nn.ReLU()           # 激活函数

    def forward(self, x):
        # -> 通过第一个隐藏层，然后应用ReLU激活函数
        x = self.relu(self.fc1(x))
        # -> 通过第二个隐藏层，然后应用ReLU激活函数
        x = self.relu(self.fc2(x))
        # -> 通过输出层，得到最终的logit
        x = self.fc3(x)
        return x
# 实例化和定义损失函数与优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备是: {device}")
model = TitanicModel(X_train_tensor.shape[1]) # 实例化模型，传入特征数量
model.to(device) # <--- 将模型的所有参数和缓冲区移动到GPU
criterion = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 随机梯度下降优化器

# 训练与评估
epochs = 2000
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    train_corrects = 0
    for i ,data in enumerate(train_loader):
        inputs, labels = data

        inputs = inputs.to(device) # <--- 移动输入数据
        labels = labels.to(device) # <--- 移动标签数据

        outputs = model(inputs)  # 前向传播

        loss = criterion(outputs, labels)  # 计算损失

        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item() * inputs.size(0)
        # 将logits转换为概率，再转换为0/1预测
        preds = torch.round(torch.sigmoid(outputs))
        train_corrects += torch.sum(preds == labels.data)

    # --- 验证阶段 ---
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():  # 验证时不需要计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            val_corrects += torch.sum(preds == labels.data)

    # --- 计算并打印每个epoch的指标 ---
    avg_train_loss = train_loss / len(train_data)
    avg_train_acc = train_corrects.double() / len(train_data)
    avg_val_loss = val_loss / len(val_data)
    avg_val_acc = val_corrects.double() / len(val_data)

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}] | '
              f'训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_acc:.4f} | '
              f'验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_acc:.4f}')

print("\n训练完成!")

