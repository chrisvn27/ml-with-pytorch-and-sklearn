import pandas as pd
from io import StringIO

csv_data = \
"""A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.1,12.0,"""

df = pd.read_csv(StringIO(csv_data))

print("Original DataFrame:")
print(df)

print(df.isnull().sum())

print(df.values)

#Stopped at page 107

print(df.dropna(axis=0))

print(df.dropna(axis=1))

print(df.dropna(how='all')) #drops rows where all elemens are NaN

#drop rows that hve fewer than 4 real values
print(df.dropna(thresh=4))

# only drop drows where NaN appear in specific columns (here: 'C')
print(df.dropna(subset=['C']))

# Method of filling missing values 'mean imputation'
from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

# Convenient method to fill missing values using pandas
print(df.fillna(df.mean()))
