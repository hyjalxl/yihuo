# coding=utf-8
# user=hu
import torch
import torch.utils.data as Data
import numpy as np
BATCH_SIZE = 10

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# print(x)
# print(y)
# 转化torch能识别的dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
print(torch_dataset)

# 把dataset放入DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # 是否按照顺序
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):

        # 这是假设训练的地方

        # 下面可以打印出要训练的数据
        print('Epoch:', epoch, '|step:', step, '|batch x:',
              batch_x.numpy(), '\n', '\t'*4, '', '|batch y:', batch_y.numpy())


if __name__ == "__main__":
    pass