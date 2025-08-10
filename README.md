# Pytorch深度学习实践

# 深度学习基础

## 损失函数 

(Cost Function / Loss Function)

评价模型的预测值和真实值不一样的程度，用来衡量模型“犯错”程度的函数，即预测值与真实值之间的差距。

**损失 (Loss)**: 通常指**单个样本**的误差。

- **公式**:![image-20250629205806483](https://gitee.com/TChangQing/qing_images/raw/master/images/20250629205806515.png)

损失函数：

+ 评价模型性能，既预测结果和真实结果越接近性能越好。
+ 参数优化，通过计算损失函数，进行反向传播，不断更新参数。

### 均方差损失函数

用于线性回归问题，计算预测值与真实值之间的平均平方差

**常用公式 (MSE - 均方误差)**:

- $$
  Cost(w,b) = \frac{1}{N} \sum_{i=1}^{N} \left(y_i' - y_i\right)^2 = \frac{1}{N} \sum_{i=1}^{N} \left(wx_i + b - y_i\right)^2
  $$

**核心思想**: **Cost 值越小，说明模型越好**。我们所有优化的目标，就是最小化这个 Cost。

### 交叉熵损失函数

**交叉熵是在分类任务中，衡量模型预测结果好坏的一种损失函数**。

它的核心思想是**衡量两个概率分布之间的差异**。 在机器学习中，这两个分布分别是：

1. **真实分布 (True Distribution)**: 这是正确答案的概率分布。在分类问题中，它是一个“one-hot”向量。例如，一个三分类问题（猫、狗、鸟），一张猫的图片其真实分布就是 `[1, 0, 0]`，表示“是猫的概率为100%，是狗的概率为0%，是鸟的概率为0%”。
2. **预测分布 (Predicted Distribution)**: 这是你的模型经过计算后，输出的每个类别的预测概率。例如，模型可能预测这张图片是 `[0.7, 0.2, 0.1]`，表示“70%的可能是猫，20%是狗，10%是鸟”。

交叉熵损失函数的作用就是计算这两个分布之间的“距离”。**如果模型的预测分布与真实分布越接近，交叉熵损失就越小；反之，如果相差越大，损失就越大**。 我们的训练目标就是通过调整模型参数，来最小化这个交叉熵损失。

![image-20250709102125104](https://gitee.com/TChangQing/qing_images/raw/master/images/20250709102125153.png)

负号的作用是**将这个负的对数损失“扳正”**，变成一个**正数**，**抵消对数函数对(0,1]区间的概率值取对数后产生的负号**，从而将损失值转化为一个我们习惯于优化的、非负的、越小越好的正数。

![image-20250709104104227](https://gitee.com/TChangQing/qing_images/raw/master/images/20250709104104292.png)



## 梯度下降 

梯度下降是一种优化算法，用于寻找函数（在这里是代价函数）的最小值。

![beab0e92-857d-4ad5-8313-5e7c6a4e8d33](https://gitee.com/TChangQing/qing_images/raw/master/images/20250810162646362.png)

- **核心比喻 (下山)**:
  1. 站在山坡任意一点（随机初始化 `w` 和 `b`）。
  2. 感受当前位置**最陡峭的下坡方向**（计算梯度）。
  3. 朝着这个方向迈出一小步（用学习率更新参数）。
  4. 不断重复，直到走到山谷最低点（代价函数的最小值）。

**2. 关键组成**

- **梯度 (Gradient / 导数)**: ![image-20250629205834731](https://gitee.com/TChangQing/qing_images/raw/master/images/20250629205834765.png)
  - **定义**: 代价函数在某一点的**斜率**，指向函数值**上升最快**的方向。
  - **作用**: 梯度的**反方向** (`-gradient`) 就是代价函数值**下降最快**的方向。
- **学习率 (Learning Rate, α)**:
  - **定义**: 每次更新参数时迈出的“步长”。
  - **作用**: 控制学习的速度。太小则收敛过慢，太大则可能在最低点附近来回“震荡”，甚至错过最低点。

**3. 更新规则 (The Update Rule)**

梯度下降算法的核心迭代公式。

- **公式**:

  ![image-20250629205848894](https://gitee.com/TChangQing/qing_images/raw/master/images/20250629205848936.png)

- **工作原理**:

  - `w` 的新值，等于 `w` 的旧值，减去 `学习率` 乘以 `w` 方向的梯度。
  - `b` 的更新同理。

**4. 梯度下降的变种**

| **特性**         | **批量梯度下降 (BGD)** | **随机梯度下降 (SGD)** | **小批量梯度下降 (Mini-batch GD)** |
| ---------------- | ---------------------- | ---------------------- | ---------------------------------- |
| **每次更新数据** | **全部**训练数据       | **1个**随机样本        | **一小批**随机样本 (e.g., 32)      |
| **优点**         | 方向准确，路径平滑     | 更新速度快             | **效率与稳定性的最佳平衡**         |
| **缺点**         | 计算开销大，慢         | 路径震荡，不稳定       | 需额外设置批大小                   |
| **现状**         | 数据量大时基本不用     | 很少单独使用           | **现代深度学习的标配**             |

## 激活函数

**类似于大脑中的神经元：** 一个神经元会接收来自成百上千个其他神经元的电信号。它把这些信号全部加起来。但它不是简单地把这个总和再传下去。它有一个“**触发阈值**”。

- 如果所有信号的总和非常微弱，低于这个阈值，这个神经元就**保持沉默**，什么也不做，我们说它“**未被激活**”。
- 如果信号总和足够强，超过了阈值，这个神经元就会“**开火**”（Fire），产生一个强烈的电脉冲，传递给下游的神经元。我们说它“**被激活了**”。

**神经网络中的激活函数：** 它扮演的就是这个“**触发机制**”或“**开关**”的角色。

- 神经元先把所有输入 `x` 和对应的权重 `w` 相乘再求和，得到一个总的输入信号 `z = Σwᵢxᵢ + b`。

- 然后，**激活函数**会接收这个总信号`z`，并“决定”这个神经元最终应该输出什么。它决定了神经元是否“开火”，以及“火力”有多猛。

  > 目的是为了通过**在架构中强制加入非线性模块（激活函数）**，赋予了它塑造非线性关系的能力

### Sigmoid

![image-20250810151219841](https://gitee.com/TChangQing/qing_images/raw/master/images/20250810151219932.png)

### Tanh

![image-20250810151533195](https://gitee.com/TChangQing/qing_images/raw/master/images/20250810151533273.png)

### ReLu

![image-20250810151718818](https://gitee.com/TChangQing/qing_images/raw/master/images/20250810151718893.png)



## 反向传播



## 线性回归(线性模型)

**1. 核心目标**

线性模型的目标是找到一个线性的、直线的函数关系，来描述输入特征 `X` 和输出标签 `y` 之间的关系。

**2. 假设函数 (Hypothesis Function)**

- **公式**:

  ![image-20250629205714017](https://gitee.com/TChangQing/qing_images/raw/master/images/20250629205714094.png)

- **参数说明**:

  - `X`: 输入特征 (e.g., 房屋面积)。
  - `y'`: 模型的预测值 (e.g., 预测的房价)。
  - `w`: **权重 (Weight)**，代表特征的重要性 (e.g., 每平米多少钱)。
  - `b`: **偏置 (Bias)**，代表模型的基准线或偏移量 (e.g., 房屋的起步价)。

- **学习目标**: 找到最优的 `w` 和 `b`，使得模型的预测值 y′ 无限接近真实值 `y`。

## 逻辑回归(分类问题)

**1. 核心目标**

在线性回归模型的基础上，使用概率模型(如**Sigmoid函数**)，将线性模型的结果压缩到[0,1]之间，使其拥有概率意义，它可以将任意输入映射到[0,1]区间，实现值到概率转换



## 处理多维特征的输入

利用矩阵的空间变化，讲高维降到低维

![image-20250709160930623](https://gitee.com/TChangQing/qing_images/raw/master/images/20250709160930773.png)

![image-20250709161006391](https://gitee.com/TChangQing/qing_images/raw/master/images/20250709161006477.png)

### 建立模型

```python
class DiabetesModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesModel, self).__init__()
        # 定义网络结构
        self.layer1 = nn.Linear(input_size, 32) # 输入层 -> 隐藏层1
        self.relu = nn.ReLU() # ReLU激活函数
        self.layer2 = nn.Linear(32, 16)        # 隐藏层1 -> 隐藏层2
        self.output_layer = nn.Linear(16, 1)   # 隐藏层2 -> 输出层

    def forward(self, x):
        # 定义前向传播路径
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
```



## 加载数据集

![image-20250709180548150](https://gitee.com/TChangQing/qing_images/raw/master/images/20250709180548235.png)

**Epoch**

一个 Epoch 指的是**整个训练数据集中的所有样本都被模型“过目”了一遍**的过程（forward+backward）

**Batch-Size**

**一次迭代（即一次参数更新）中所用到的样本数量**。

**Batch-Size 越大**: 每次更新时考虑的样本更多，梯度方向更准确、稳定；能更好地利用硬件的并行计算能力，每个Epoch的训练时间更短。但需要更多内存。

**Batch-Size 越小**: 引入的随机性更大，训练过程更“震荡”，有时反而有助于模型跳出局部最优解；内存占用小。但训练时间可能更长。

**Iterations**

完成一个 Epoch所需要的**批次数量**，也等于一个Epoch中模型**参数更新的次数**。![image-20250709181234308](https://gitee.com/TChangQing/qing_images/raw/master/images/20250709181234353.png)

**完整的例子**

- **总样本数 (Total Samples)**: 8000
- **批量大小 (Batch-Size)**: 100
- **轮次数 (Epochs)**: 10

我们可以计算出：

- **每个Epoch的迭代次数 (Iterations per Epoch)**: `8000 / 100 = 80` 次
- **总迭代次数 (Total Iterations)**: `10 Epochs * 80 Iterations/Epoch = 800` 次

### Dataset和DataLoader

```python
class DiabetesDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

dataset = DiabetesDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
```

**1. `def __init__(self):` (构造函数初始化)**

在创建数据集实例时调用一次，例如 `dataset = DiabetesDataset()`。负责所有**一次性**的准备工作。

加载数据文件的路径。**将所有数据一次性读入内存**。计算并存储数据集的总长度

**2. `def __len__(self):` (数据总数)**

 `DataLoader`在初始化时，以及在每个epoch开始前，会调用这个方法。必须返回一个整数，代表这个数据集中**样本的总数量**。通常就是简单地返回在 `__init__` 中计算好的 `self.len`。

**3. `def __getitem__(self, index):` (根据索引取出数据)**

 `DataLoader`在构建一个批次（batch）时，会频繁地、逐个地调用这个方法。根据传入的**索引（index）`idx`**，准确地获取并返回**一个样本**的数据及其对应的标签。

- 从 `self.features` 中根据 `index` 取出第 `index` 个样本的特征。
- 从 `self.labels` 中根据 `index` 取出第 `index` 个样本的标签。
- 将它们作为一个元组 `(feature, label)` 返回。

**`DataLoader` 的核心参数：**

- **`dataset=dataset`**: 告诉叉车要去哪个仓库工作。这里就是我们刚刚实例化的`dataset`对象。
- **`batch_size=32`**: 定义一次有多少min-batch。这里它会一次性从数据集中取出32个样本，并将它们打包成一个批次（batch）。
- **`shuffle=True`**: `True`表示在**每个epoch开始前**，索引完全打乱，可以有效防止模型学习到数据的排列顺序，增强模型的泛化能力。在训练时通常设置为`True`，在测试时则设置为`False`。
- **`num_workers=2`**: 这是性能优化的关键参数，代表**使用多少个子进程来预加载数据**。
  - 如果 `num_workers=0` (默认)，数据加载会在主进程中进行。当GPU在训练当前批次时，CPU就在“休息”。
  - 如果 `num_workers=2`，PyTorch会启动**2个额外的进程**。当GPU在忙于训练第N个批次时，这两个“工人”已经在后台马不停蹄地准备第N+1、N+2个批次的数据了。这样一来，GPU训练完后无需等待，可以直接拿到新数据开始下一轮计算，极大地**提高了训练效率**。

## 多分类问题(Softmax 分类器)

### Softmax

![image-20250707103040900](https://gitee.com/TChangQing/qing_images/raw/master/images/20250707103041041.png)

### Softmax 函数

**Sigmoid** 函数是将一个数压缩到 (0, 1) 区间，而 **Softmax** 函数则是将**一组数**进行同样的操作，并且让它们的**总和为1**。

- **作用**: Softmax 接收一组任意的实数（logits），并将它们转换成一个**概率分布**。
- **特性**:
  1. **大于0**: 输出的每个数值都在 (0, 1) 区间内。
  2. **和为1**: 所有输出数值的总和等于1。



### NLLLoss

![image-20250707103253135](https://gitee.com/TChangQing/qing_images/raw/master/images/20250707103253230.png)

**One-Hot编码** 描述的是**单个样本的真实标签**属于哪个**类别**。它是一个长度等于类别总数的向量，其中，**正确类别的位置为1，其余所有位置为0**。One-hot编码提供了一个与模型输出的概率分布（如 `[0.7, 0.2, 0.1]`）格式完全对应的**真实概率分布**，从而可以计算它们之间的交叉熵损失。

**NLLLoss** (Negative Log Likelihood Loss, 负对数似然损失)

从经过Softmax 计算后的一组对数概率中，**“挑出”真实标签所对应的那一个对数概率，再给它取个负号**，就得到了最终的损失。

**例子**:

- 真实标签: `2` (One-hot: `[1, 0, 0]`)
- 模型的Softmax输出: `[0.38 0.34 0.28]`
- `NLLLoss` 会挑出索引为 `1` 的值 `-0.97`，然后取负号，得到最终 `loss = 0.97`。

![image-20250707103439075](https://gitee.com/TChangQing/qing_images/raw/master/images/20250707103439175.png)

Pytorch中的CrossEntrepyLoss

![image-20250707103526859](https://gitee.com/TChangQing/qing_images/raw/master/images/20250707103526972.png)

**`nn.CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`**

1. 将模型的**原始、未经任何激活函数处理的 logits** 直接喂给它。
2. 将**非One-hot形式的、整数的真实标签**（例如 `2`）也喂给它。
3. 它会在内部自动帮你完成 `Log-Softmax` 的计算，以及 `NLLLoss` 的挑选和取负操作。

## 特征缩放

```python
# 使用 train_test_split 进行划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 特征缩放: 在训练集上fit，然后在训练集和验证集上transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val) # 注意：验证集只用transform
```

**将所有数值型特征放在一个公平的起跑线上**，以帮助模型更好地学习

**1. 为什么需要特征缩放？(Why?)**

- **`Age`**: 数值范围大约在 0 到 80 之间。
- **`Pclass` (船舱等级)**: 数值范围是 1, 2, 3。

对于一个模型来说，它看到`Age`的数值（比如40）远大于`Pclass`的数值（比如2）。如果不对它们进行处理，模型可能会错误地认为`Age`这个特征比`Pclass`重要得多，仅仅因为它数值大。

**特征缩放的目的**就是消除这种由数值范围带来的“偏见”，将所有特征都转换到一个相似的、较小的尺度上。这能让梯度下降等优化算法工作得更稳定、收敛得更快。

**2. `StandardScaler` 是做什么的？(What?)**

`StandardScaler`是`sklearn`库中提供的一种特征缩放方法，它的策略是**标准化 (Standardization)**。

它会把每一列特征的数据都转换成**均值为0，标准差为1**的分布。这就像把每个班级的考试成绩都转换成标准分，这样就可以公平地比较不同班级的学生表现了。

**3. `fit_transform` 和 `transform` 的区别 (How?)**

这是最关键、也最容易混淆的地方。我们可以用一个“**制作模具**”的比喻来理解：

- **`scaler.fit(X_train)`**: 这是**学习**或**制作模具**的步骤。程序会**只分析训练集 `X_train`**，计算出训练集中每一列特征的平均值（μ）和标准差（σ）。它把这些计算出来的“规则”保存在`scaler`这个对象里。这就好比根据一块泥土（训练集）制作了一个模具。
- **`scaler.transform(X)`**: 这是**应用**或**使用模具**的步骤。它会使用**已经学习到的**平均值和标准差，来转换任何给定的数据。它不会再计算新的平均值或标准差。这就好比用已经做好的模具去塑造新的泥土。
- **`scaler.fit_transform(X_train)`**: 这是一个方便的快捷方式，它把上面两个步骤合二为一，**只对训练集使用**。它先在`X_train`上学习（fit），然后立刻用学到的规则来转换`X_train`（transform）。

**4. 为什么验证集只能用 `transform`？**

**这是为了模拟真实世界，防止“数据泄露 (Data Leakage)”。**

- **验证集/测试集** 的作用是模拟模型在未来遇到的**全新的、未知的数据**。
- 在现实世界中，我们不可能提前知道未来数据的平均值和标准差。我们唯一拥有的信息就是我们手上的**训练集**。
- 因此，我们必须用**从训练集中学到的“规则”（即平均值和标准差）**来处理验证集。我们假装对验证集一无所知，只能用旧的模具来塑造它。

**如果对验证集也使用 `fit_transform` 会发生什么？** 那就意味着我们的模型在训练阶段，就已经“偷看”了验证集的数据分布（知道了验证集的平均值和标准差）。这会让模型在验证集上的表现看起来过于乐观，从而导致我们对模型的泛化能力做出错误的评估。

- **`fit_transform()`**: **只对训练集使用**，让缩放器学习并转换训练数据。
- **`transform()`**: **对验证集和测试集使用**，用从训练集学到的规则来转换新数据。

# 全连接神经网络

神经网络的本质是**寻找非线性的空间变换函数**

![image-20250808155144807](https://gitee.com/TChangQing/qing_images/raw/master/images/20250808155144890.png)



# 卷积神经网络(CNN)

![image-20250715115138518](https://gitee.com/TChangQing/qing_images/raw/master/images/20250715115138633.png)

卷积层

池化层



LeNet

AlexNet

# 循环神经网络(RNN)

