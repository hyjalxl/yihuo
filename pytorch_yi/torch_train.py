# coding=utf-8
# user=hu
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 1000000), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承父类__init__功能
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 正向传播输入值，神经网络分析输出值
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

# 传入net的所有参数，学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()
# plt.ion()


for t in range(100):
    # 喂给net训练集数据x,输出预测值
    prediction = net(x)
    # 计算两者的误差
    loss = loss_func(prediction, y)
    # 清空上一部的残余更新参数
    optimizer.zero_grad()
    # 误差反向传播，计算参数更新值
    loss.backward()
    # 将参数更新值施加到Net的parameters上
    optimizer.step()
    # print(loss.data[0], prediction.data.numpy()[0])
    # 下面是画图模块
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(1)
plt.ioff()
plt.show()

if __name__ == '__main__':
    pass
