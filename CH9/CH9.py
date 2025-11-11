import pandas as pd

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 
           'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

print(df.head())
print(df.shape)

#Changing string type to int
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

print(df.isnull().sum())

df = df.dropna(axis=0)
print(df.isnull().sum())

#Stopped at page 274
