import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from torch.autograd import Variable
from sklearn.datasets import load_breast_cancer


X = load_breast_cancer().data
Y = load_breast_cancer().target

class SVM(torch.nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = torch.nn.Linear(30, 2)  # Implement the Linear function input30 output=2
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd

net = SVM()     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

X = torch.FloatTensor(X)  
Y = torch.FloatTensor(Y)

for t in range(200):
    out = net(X.float())
    loss = loss_func(out, Y.long())
    
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gr
    
    if t % 2 == 0:
        # plot and show learning process
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = Y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        
        print("Epoch {}, Loss: {}".format(accuracy, loss))
