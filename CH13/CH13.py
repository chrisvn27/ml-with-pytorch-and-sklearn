import torch

# Creating a graph in Pytorch
def compute_z(a ,b ,c):
    r1 = torch.sub(a,b)
    r2 = torch.mul(r1,2)
    z = torch.add(r2, c)
    return z

print('Scalar (Rank 0) inputs: ', compute_z(torch.tensor(1), torch.tensor(2), torch.tensor(3)))

print('Rank 1 inputs: ', compute_z(torch.tensor([1]), torch.tensor([2]), torch.tensor([3])))

print('Rank 2 inputs: ', compute_z(torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[3]])))

a = torch.tensor(3.14, requires_grad=True)
print(a)

b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(b)

w = torch.tensor([1.0, 2.0, 3.0])
print(w.requires_grad)

w.requires_grad_()
print(w.requires_grad)

import torch.nn as nn
torch.manual_seed(1)
w = torch.empty(2,3)
nn.init.xavier_normal_(w)
print(w)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2,3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1,2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)
    
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = torch.add(torch.mul(w, x), b)
loss = (y-z).pow(2).sum()

loss.backward()
print('dL/dw: ', w.grad)
print('dL/db: ', b.grad)

#verifying the computed gradient
print(2* x * ((w * x + b) - y))
print(2*1*((w * x + b)-y))

#Stopped at pg 417

model = nn.Sequential(
    nn.Linear(4,16),
    nn.ReLU(),
    nn.Linear(16,32),
    nn.ReLU()
)
print(model)

nn.init.xavier_uniform_(model[0].weight) #modifying first layer of sequential nn.Linear(4,16)
l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum() #second layer

print(model[2].weight.shape, model[2].weight.abs().sum().shape)

loss_fn = nn.BCELoss() #binary entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


#Solving an XOR classification problem

import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(1)
np.random.seed(1)
x = np.random.uniform(low=-1, high=1, size=(200,2))
y= np.ones(len(x))
y[x[:,0]*x[:,1]<0]=0
n_train = 100
x_train = torch.tensor(x[:n_train,:], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:,:], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype= torch.float32)
fig = plt.figure(figsize=(6,6))
plt.plot(x[y==0,0], x[y==0,1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1,0], x[y==1,1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=15)
plt.ylabel(r'$x_2$', size=15)
plt.show()

model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)
print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

from torch.utils.data import DataLoader, TensorDataset
train_ds = TensorDataset(x_train, y_train)
batch_size = 2
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

torch.manual_seed(1)
num_epochs=200
def train(model, num_epochs, train_dl, x_valid, y_valid):
    loss_hist_train = [0]*num_epochs
    accuracy_hist_train = [0]*num_epochs
    loss_hist_valid = [0]*num_epochs
    accuracy_hist_valid = [0]*num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:,0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred>=0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()
        loss_hist_train[epoch] /= n_train/batch_size #n_train/batch_size = # of batches
        accuracy_hist_train[epoch] /= n_train/batch_size #n_train/batch_size = # of batches
        pred = model(x_valid)[:,0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred>=0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()
        if epoch % 10 == 0 :
            print(f"Epoch {epoch} , loss_train: {loss_hist_train[epoch]}, loss_valid: {loss_hist_valid[epoch]}")
    
    return loss_hist_train, loss_hist_valid, \
            accuracy_hist_train, accuracy_hist_valid

history = train(model, num_epochs, train_dl, x_valid, y_valid)

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw=4)
plt.plot(history[1], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size = 15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw=4)
plt.plot(history[3], lw=4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)
plt.show()

model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
print(model)

history = train(model, num_epochs, train_dl, x_valid, y_valid)
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw=4)
plt.plot(history[1], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size = 15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw=4)
plt.plot(history[3], lw=4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)
plt.show()
