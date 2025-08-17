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