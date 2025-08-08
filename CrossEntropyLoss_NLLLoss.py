import  numpy as np
import  torch

y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])

#softmax function
y_pred = np.exp(z)/np.sum(np.exp(z))
print(y_pred.round(2))
# NLLLoss
loss = (-y * np.log(y_pred)).sum()

print("softmax + NLLLoss：" + str(loss.round(2)))

y_1 = torch.LongTensor([0]) # target class index

z_1 = torch.Tensor([0.2, 0.1, -0.1]) #softmax output

criterion = torch.nn.CrossEntropyLoss() #  CrossEntropyLoss combines softmax and NLLLoss

loss_1 = criterion(z_1.view(1,-1), y_1) # .View表示将z_1变成一个2维的tensor,其中第一维为1，第二维为-1（自动计算）

print("CrossEntropyLoss：" + str(round(loss_1.item(),2)))
