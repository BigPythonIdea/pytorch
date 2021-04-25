import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from sklearn.datasets import load_breast_cancer


X = load_breast_cancer().data
Y = load_breast_cancer().target

plt.scatter(x=X[:, 0], y=X[:, 1])

class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(30, 10)  # Implement the Linear function input30 hiidend10
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd

data = X  # Before feature scaling
X = (X - X.mean())/X.std()  # Feature scaling
Y[Y == 0] = -1  # Replace zeros with -1
plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling


learning_rate = 0.2  # Learning rate
epoch = 1000  # Number of epochs
batch_size = 1  # Batch size

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500

model = SVM()  # Our model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
for epoch in range(epoch):
    perm = torch.randperm(N)  # Generate a set of random numbers of length: sample size
    sum_loss = 0  # Loss for each epoch
        
    for i in range(0, N, batch_size):
        x = X[perm[i:i + batch_size]]  # Pick random samples by iterating over random permutation
        y = Y[perm[i:i + batch_size]]  # Pick the correlating class
        
        x = Variable(x)  # Convert features and classes to variables
        y = Variable(y)

        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(x)  # Compute the output by doing a forward pass
        
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        sum_loss += loss.data.numpy()  # Add the loss
        
    print("Epoch {}, Loss: {}".format(epoch, sum_loss))
