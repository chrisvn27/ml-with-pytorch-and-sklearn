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