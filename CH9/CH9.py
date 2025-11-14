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

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
scatterplotmatrix(df.values, figsize=(12,10),
                  names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()

import numpy as np
from mlxtend.plotting import heatmap

cm = np.corrcoef(df.values.T)
print(cm.shape)
hm = heatmap(cm, row_names= df.columns, column_names= df.columns)
plt.tight_layout()
plt.show()


class LinearRegressionGD:
    def __init__(self, eta= 0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return self.net_input(X)
    
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()
lr = LinearRegressionGD(eta= 0.1)
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter + 1), lr.losses_)
plt.xlabel('MSE')
plt.ylabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

lin_regplot(X_std, y_std, lr)
plt.xlabel('Liven area above ground standardized')
plt.ylabel('Saleprice (standardized)')
plt.show()


feature_std = sc_x.transform(np.array([[2500]]))
target_std = lr.predict(feature_std)
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f"Sales price: ${target_reverted.flatten()[0]:.2f}")

print(f"Slope: {lr.w_[0]:.3f}")
print(f"Intercept (bias): {lr.b_[0]:.3f}")

# Stopped at beginning of page 283

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)
y_pred = slr.predict(X)
print(f"Slope: {slr.coef_[0]:.3f}")
print(f"Intercept: {slr.intercept_:.3f}")

lin_regplot(X,y, slr)
plt.xlabel("Living area above ground in square feet")
plt.ylabel("Sale price in U.S. dollars")
plt.tight_layout()
plt.show()

# Analytical solution: w = (X^TX)**-1 X^T y
# adding a column vecotr of "ones"
Xb = np.hstack((np.ones((X.shape[0],1)),X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print(f"Slope: {w[1]:.3f}")
print(f"Intercept: {w[0]:.3f}")


from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100,
    min_samples=0.95,
    residual_threshold=None,
    random_state=123
)
ransac.fit(X,y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3,10,1)
line_y_ransac= ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f"Slope: {ransac.estimator_.coef_[0]:.3f}")
print(f"Intercept: {ransac.estimator_.intercept_:.3f}")

def mean_absolute_deviation(data):
    return np.mean(np.abs(data- np.mean(data)))

print(mean_absolute_deviation(y))

#Stopped at page 288
