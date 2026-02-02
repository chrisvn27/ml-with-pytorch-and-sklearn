import torch
from torch_geometric.datasets import QM9
from  torch_geometric.loader import DataLoader
from torch_geometric.nn  import NNConv, global_add_pool
import torch.nn.functional  as F
import torch.nn as nn
import numpy as np

#let's load the  QM9 small moelcule dataset
dset = QM9('.')
print(len(dset))
# Here's how torch geometric  wraps data
data = dset[0]
print(data)
# can access attributes  directly
print(data.z)
# the  atomic  number of each atom  can add attributes
data.new_attribute =  torch.tensor([1, 2, 3])
print(data)
# can move all attributes between devices
device = ("cuda"  if torch.cuda.is_available() else "cpu")
# data.to(device)
# print(data.new_attribute.is_cuda)
print(data.y)

class ExampleNet(torch.nn.Module):
    def __init__(self,  num_node_features, num_edge_features):
        super().__init__()
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_node_features*32))
        
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32*16))
        
        self.conv1  = NNConv(num_node_features, 32,  conv1_net)
        self.conv2  = NNConv(32, 16, conv2_net)
        self.fc_1 = nn.Linear(16, 32)
        self.out =  nn.Linear(32, 1)

    def forward(self, data):
        batch, x, edge_index, edge_attr =\
        (data.batch, data.x, data.edge_index,  data.edge_attr)
        # First graph conv layer
        x  = F.relu(self.conv1(x, edge_index, edge_attr))
        # Second graph conv layer
        x  = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        x  = F.relu(self.fc_1(x))
        output = self.out(x)
        return output
    
from torch.utils.data import random_split
train_set, valid_set, test_set  = random_split(dset, [110000, 10831, 10000])
trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
validloader = DataLoader(valid_set, batch_size=32, shuffle=True)
testloader =  DataLoader(test_set, batch_size=32, shuffle=True)

# initialize a network
qm9_node_feats , qm9_edge_feats = 11, 4
net = ExampleNet(qm9_node_feats, qm9_edge_feats)

# initialize an optimizer with some reasonable parameters
optimizer = torch.optim.Adam(net.parameters(),  lr= 0.01)
epochs  = 5
target_idx = 1 #index position of the polarizability label
net.to(device)

for total_epochs in range(epochs):
    epoch_loss = 0
    total_graphs = 0
    net.train()
    for batch in trainloader:
        batch.to(device)
        optimizer.zero_grad()
        output = net(batch)
        loss  = F.mse_loss(
            output, batch.y[:, target_idx].unsqueeze(1))
        epoch_loss += loss.item()
        total_graphs += batch.num_graphs
        loss.backward()
        optimizer.step()
    train_avg_loss  = epoch_loss /  total_graphs
    val_loss = 0
    total_graphs  = 0
    net.eval()
    for batch  in validloader:
        batch.to(device)
        output = net(batch)
        output = net(batch)
        loss = F.mse_loss(
            output, batch.y[:, target_idx].unsqueeze(1))
        val_loss +=  loss.item()
        total_graphs +=  batch.num_graphs
    val_avg_loss = val_loss / total_graphs
    print(f"Epochs: {total_epochs} | "
          f"epoch avg. loss: {train_avg_loss:.2f} | "
          f"validation avg.loss: {val_avg_loss:.2f}")
    
net.eval()
predictions = []
real = []
for batch in testloader:
    output  = net(batch.to(device))
    predictions.append(output.detach().cpu().numpy())
    real.append(batch.y[:, target_idx].detach().cpu().numpy())
real =  np.concatenate(real)
predictions = np.concatenate(predictions)

import matplotlib.pyplot as plt
plt.scatter(real[:500], predictions[:500])
plt.xlabel('Isotropic polarizability')
plt.ylabel('Predicted isotropic polarizability')
plt.show()