
# Hello pytorch

先確認pytorch GPU安裝成功
```
import torch
from torch.autograd import Variable
```
套用基本tensor
```
tensor = torch.FloatTensor([[1,2],[3,4]])
print(tensor)
```
確認GPU
```
variable = Variable(tensor, requires_grad=True)
print(variable)
```
顯示True表GPU安裝成功

小試身手 【x^2】
---
```
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)
```

但是時刻記住，變量計算時，它在背景幕布後面一步步默默地建造著一個龐大的系統，叫做計算圖，計算圖。這個圖是用來幹嘛的？原來是將所有的計算步驟（Node）都連接起來，最後進行誤差反向傳遞的時候，一次性將所有變量裡面的修改幅度（梯度）都計算出來，而tensor就沒有這個能力啦。

倒傳遞也輕鬆簡單
---
```
v_out.backward()
print(variable.grad
```
轉成numpy 也意外的簡單
```
print(variable.data.numpy())
```

補充一些Fuction 原理
---
Autograd: 自动微分
　　autograd包是PyTorch中神经网络的核心, 它可以为基于tensor的的所有操作提供自动微分的功能, 这是一个逐个运行的框架, 意味着反向传播是根据你的代码来运行的, 并且每一次的迭代运行都可能不同.
  
![](https://i.imgur.com/cwuaDZQ.png)

![](https://i.imgur.com/gQnjkUb.png)


Torch Activation Function
---
```
import torch.nn.functional as F
from torch.autograd import Variable
```

```
x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()
```

```
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
```
![](https://i.imgur.com/9ZMYrad.png)

plot一下...
```
import matplotlib.pyplot as plt

plt.figure(1, figsize=(8,6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2,6))
plt.legend(loc='best')

plt.show()
```
![](https://i.imgur.com/1ocC8Gn.png)
