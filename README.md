###### tags: `Pytorch`


# 回歸



### 魔改一下 乳癌分析 使用pytorch 在cancer.py，代碼在下面差不多
#### 須注意 在向前傳遞是double還是float還有Lone問題



```
x.to(torch.float64)
y.to(torch.float64)
```
```
out = net(x.float())             # input x and predict based on x
loss = loss_func(out, y.long())     # must be (1. nn output, 2. target), the target label is NOT one-hotted
```



> 解決方案 : https://www.extutorial.com/ask/352248






建立數據
---
```
import torch
import matplotlib.pyplot as plt


# y = a * x^2 + b
x = torch.unsqueeze(torch.linspace(-1, 1), dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())


plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

```

建立網路
---
```
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:18:39 2021

@author: Mooncat
"""

import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1), dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())



class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #繼承 init 的功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        
    
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
        

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5) #建立優化器
loss_func = torch.nn.MSELoss()                        

for t in range(100):
    prediction = net(x)
    
    loss = loss_func(prediction, y)
    
    optimizer.zero_grad() #梯度降為0
    loss.backward() #反向傳遞
    optimizer.step() #每0.5步優化梯度
    
    #視覺化搭建流程
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
```


