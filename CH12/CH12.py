import torch 
import numpy as np
np.set_printoptions(precision=3)
a = [1, 2, 3]
b = np.array([4, 5, 6], dtype=np.int32)
t_a = torch.tensor(a)
t_b = torch.from_numpy(b)
print(t_a)
print(t_b)
t_ones = torch.ones(2, 3)
print(t_ones.shape, t_ones)
rand_tensor = torch.rand(2, 3)
print(rand_tensor)
t_a_new = t_a.to(torch.int64)
print(t_a_new.dtype)
t = torch.rand(3, 5)
t_tr = torch.transpose(t, 0 , 1)
print(t.shape, '---> ', t_tr.shape)
t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t_reshape.shape)
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t,2)
print(t.shape, '---> ', t_sqz.shape)

torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5,2))

#element wise multiplication
t3 = torch.multiply(t1,t2)
print(t3)

#mean along the rows
t4 = torch.mean(t1, axis=0)
print(t4)

#Matrix mult
t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)
t6 = torch.matmul(torch.transpose(t1, 0 ,1), t2)

#Calculating the L^2 norm
norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)
print(np.sqrt(np.sum(np.square(t1.numpy()), axis=1)))

#Split, stack, and concatenate tensors
torch.manual_seed(1)
t = torch.rand(6)
print(t)
t_splits = torch.chunk(t, 3)
print([item.numpy() for item in t_splits])

torch.manual_seed(1)
t = torch.rand(5)
print(t)
t_splits = torch.split(t, split_size_or_sections=[3,2])
print([item.numpy() for item in t_splits])

A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A,B], axis=0)
print(C)

A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A,B], axis= 1)
print(S)

#Stopped at page 378
from torch.utils.data import DataLoader
t = torch.arange(6,dtype=torch.float32)
data_loader = DataLoader(t)

for item in data_loader:
    print(item)

data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f"batch {i}: ", batch)

for batch in data_loader:
    print(batch)

torch.manual_seed(1)
t_x = torch.rand([4,3], dtype=torch.float32)
t_y = torch.arange(4)

from torch.utils.data import Dataset
class JointDataset(Dataset):
    def __init__(self,x , y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

torch.manual_seed(1)    
joint_dataset = JointDataset(t_x, t_y)

for example in joint_dataset:
    print('x: ',example[0], '  y: ', example[1])

torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
for i, batch in enumerate(data_loader, 1):
    print(f"batch {i}: x : {batch[0]}, \n y: {batch[1]}")

for epoch in range(2):
    print(f"epoch {epoch + 1}")
    for i, batch in enumerate(data_loader,1):
        print(f"batch {i}: x : {batch[0]}, \n y: {batch[1]}")

#Stopped at page 382

import pathlib

import matplotlib.pyplot as plt
import os
from PIL import Image

import torchvision

image_path = './'
celeba_dataset = torchvision.datasets.CelebA(
    image_path, split='train', target_type='attr', download=True
)

assert isinstance(celeba_dataset, torch.utils.data.Dataset)

example = next(iter(celeba_dataset))
print(example)

from itertools import islice
fig = plt.figure(figsize=(12,8))
for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{attributes[31]}', size=15)
plt.show()

mnist_dataset = torchvision.datasets.MNIST(image_path, 'train', download=False)
assert isinstance(mnist_dataset, torch.utils.data.Dataset)
example = next(iter(mnist_dataset))
print(example)

fig = plt.figure(figsize=(15,6))
for i, (image,label) in islice(enumerate(mnist_dataset), 10):
    ax = fig.add_subplot(2,5,i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image, cmap='gray_r')
    ax.set_title(f"{label}", size=15)
plt.show()

##Building an NN model in Pytorch
X_train = np.arange(10, dtype='float32').reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0,
                    6.3, 6.6, 7.4, 8.0,
                    9.0],dtype='float32')
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Standardize the features
from torch.utils.data import TensorDataset
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle= True)

torch.manual_seed(1)
weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)

def model(xb):
    return xb @ weight + bias # z= xw + b

def loss_fn(input, target):
    return (input - target).pow(2).mean()

learning_rate = 0.001
num_epochs = 200
log_epochs = 10

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()

    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate
        weight.grad.zero_()
        bias.grad.zero_()
    if epoch % log_epochs == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

print("Final Parameters: ", weight.item(), bias.item())

X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1,1)
X_test_norm = (X_test - X_test.mean()) / X_test.std()
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm).detach().numpy()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_train_norm, y_train, 'o', markersize= 10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training examplesm', 'Linear reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis = 'both', which='major', labelsize=15)
plt.show()

import torch.nn as nn
loss_fn = nn.MSELoss(reduction='mean')
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Generate predictions
        pred = model(x_batch)
        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)
        # 3. Compute gradients
        loss.backward()
        # 4. Update parameters using gradients
        optimizer.step()
        # 5. Reset the gradients to zero
        optimizer.zero_grad()
    
    if epoch % log_epochs == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

print('Final parameters w , b: ', model.weight.item(), model.bias.item())


#Building a multilayer perceptron for cassifying flowrs in the Iris
# dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1.0/3, random_state=1
)

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3
model = Model(input_size, hidden_size, output_size)

learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

num_epochs = 100
loss_hist= [0] * num_epochs
accuracy_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[epoch] += loss.item() * y_batch.size(0) #Loss is given as mean so we need the complete values 
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist[epoch] += is_correct.sum().item()
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major',labelsize=15)
ax = fig.add_subplot(1,2,2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch',size=15)
ax.tick_params(axis='both', which='major',labelsize=15)
plt.show()

#Stopped at 398
