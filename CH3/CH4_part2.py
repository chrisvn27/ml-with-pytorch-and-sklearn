#Handling categorical data

import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
    ])
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

#Mapping ordinal features
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size']= df['size'].map(size_mapping)
print(df)

# Inverse mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))

#Encoding class labels
import numpy as np
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# Inverse mapping for class labels
inv_class_mapping = {v: k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

# Alternatively, using scikit-learn's LabelEncoder
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

#Inverse transformation using LabelEncoder
print(class_le.inverse_transform(y))

#Stopped at page 113

#Performing one-hot encoding on nominal features

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

#Implementing one-hot encoding with sklearn's OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray())

from sklearn.compose import ColumnTransformer
X = df[['color','size','price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1,2])
])
print(c_transf.fit_transform(X).astype(float))

#Even more simple is using pandas get_dummies
print(pd.get_dummies(df[['price','color','size']]))

#Dropping one dummy variable to avoid the dummy variable trap
print(pd.get_dummies(df[['price','color','size']],drop_first=True))

#Doing the same with OneHotEncoder
color_ohe = OneHotEncoder(categories='auto', drop= 'first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1,2])
])

print(c_transf.fit_transform(X).astype(float))

# Encoding odrinal features
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])

df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

df['x > M'] = df['size'].apply( lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply( lambda x: 1 if x == 'XL' else 0)

del df['size']
print(df)

#partitioning data into training and test sets
df_wine = pd.read_csv(
'https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',
header=None
)

df_wine.columns = ['Class label', 'Alcohol',
              'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium',
              'Total phenols', 'Flavanoids',
              'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue',
              'OD280/0D315 of diluted wines',
              'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

#spltting the dataset 
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test =\
      train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


#Stopped at page 119

#Using min-max scaling, a way to normalize features
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

#Standardization can be more practical for many machine learning algorithms,
# escpecially for optimization algorithms such as gradient descent

ex = np.array([0,1 ,2 ,3 ,4 ,5])
print('standardized:', (ex - ex.mean()) / ex.std())

print('normalized:', (ex - ex.min())/ (ex.max() - ex.min()))

#Using sklearn
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# L1 and L2 regularization
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l1', C= 1.0, solver= 'liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))

print('Test accuracy:', lr.score(X_test_std, y_test))

print(lr.intercept_)
print(lr.coef_)

import matplotlib.pyplot as plt
fit = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights , params = [],[]
for c in np.arange(-4.,6.0):
    lr = LogisticRegression(penalty='l1', C=10.**c,
                            solver='liblinear', multi_class='ovr', random_state=0)

    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]),colors):
    plt.plot(params, weights[:, column],
             label = df_wine.columns[column+1],
             color= color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Implementing sequential backward selection (SBS)

class SBS():
    def __init__(self, estimator, k_features,
                 scoring = accuracy_score,
                 test_size = 0.25, random_state =1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size= self.test_size,
                         random_state = self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r= dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        
        self.k_score_ = self.scores_[-1]

        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features= 1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))
