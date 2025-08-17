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
