# coding=utf-8
# user=hu
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake dataset 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
print(x.numpy())
print(y.numpy())

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 构建dataloader
torch_dataset = Data.tensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


# 默认的network形式
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Liner(1, 20)  # hidden layer
        self.predict = torch.nn.Liner(20, 1)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation for hidden layer
        x = self.predict(x)  # linear output

# 为每一种优化器创建一个net
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]


if __name__ == "__main__":
    pass