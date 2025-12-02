from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1,return_X_y=True) 
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)

#Normalize to from 0 to 255 to -1 to 1
X = ((X / 255.) - 0.5) * 2

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols= 5,
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X[y == i][5].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()

for i in range(25):
    img = X[y==7][i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000,
    random_state=123, stratify=y_temp
)

# Implementing NN with only one hidden layer

import numpy as np

def sigmoid(z): #Activation function
    return 1.0 / (1.0 + np.exp(-z))

def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0]), num_labels)
    for i, val in enumerate(y):
        ary[i,val]=1

    #Here we are making each y value into a vector with 9 dimensions. It has 1
    #on the column position where it is labelled as that number
    #i.e. ary[234, 2] = [0,0,1,0,0,0,0,0,0]
    return ary

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden,
                 num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        #Start generating weights
        rng = np.random.RandomState(random_seed)

        #hidden
        self.weight_h = rng.normal(
            loc =0.0, scale=0.1, size=(num_hidden, num_features)
        )
        self.bias_h = np.zeros(num_hidden)

        #output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden)
        )
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        #Hidden layer

        # input dim:  [n_examples, n_features]
        #       dot  [n_hidden, n_features]. T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden]
        #       dot [n_hidden,n_classes].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_out, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out
    
    def backward(self, x, a_h, a_out, y):
        ##################
        ## Output Layer weights
        ##################

        # one-hot encoding
        y_onehot= int_to_onehot(y, self.num_classes)

        # Part 1: 
        d_loss__d_a_out = 2.0*(a_out - y_onehot)/ y.shape[0]
        #Remember that here a_out is the same as y_hat, and the line above
        # is the MSE loss function derived by y_hat or in this case a_out

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z__out = a_out * (1.0 - a_out) #Sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z__out

        # gradient for output weights

        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h #Remember that Z = XW.T + b -> dZ/dW = X, where X in this case is a_h

        # input dim: [n_classes, n_examples]
        #       dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Part 2

        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out

        # output dim: [n_examples, h_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1.0 - a_h) #sigmoid derivative

        # [n_examples, b_features]
        d_z_h__d_w_h = x

        #output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,
                               d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis = 0)

        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)
    

#Stopped at page 350

