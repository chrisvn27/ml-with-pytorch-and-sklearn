from sklearn import datasets

import numpy as np

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y= iris.target

print('Class Labels: ', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify=y)


print('Labels counts in y: ', np.bincount(y))
print('Labels counts in y_train: ', np.bincount(y_train))
print('Labels counts in y_test: ', np.bincount(y_test))

from sklearn.preprocessing import StandardScaler

