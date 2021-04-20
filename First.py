# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:24:41 2021

@author: Mooncat
"""

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
print(tensor)

variable = Variable(tensor, requires_grad=True)
print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()
print(variable.grad)

# to numpy
print(variable.data.numpy())

import torch.nn.functional as F
from torch.autograd import Variable

x = torch.linspace(-5, 5, 200)
x = Variable(x)

x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

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


