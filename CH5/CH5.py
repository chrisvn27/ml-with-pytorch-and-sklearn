#Unsupervides dimensionality reduction via principal component analysis

#Principal component analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv(
'https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',
header=None
)

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = \
 train_test_split(X,y, test_size=0.3, stratify=y, random_state=0)

#Standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', eigen_vals)
#print('\nEigenvectors \n', eigen_vecs)

#Stopped at page 144

# Variance ratio = lambda_j / Sumj=1_to_d(lambda j)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse= True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, align='center', label = 'Individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label = 'Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Feture Transformation
#Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key= lambda k: k[0], reverse= True) #k[0] is sorting about the first input of the tupple : eigenvalue

W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

print('Matrix W: \n', W)

#Reducing dimentionality: x' = xW (x' = 1x2, x=1x13, W=13x2)
X_train_pca = X_train_std.dot(W)

colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label = f'Class {l}', marker = m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()

#Stopped at page 149
#Using scikit learn
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx= None, resolution =0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c = colors[idx],
                    marker=markers[idx],
                    label=f"Class {cl}",
                    edgecolor='black')
        
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# Initializing the PCA transformer and logistic regression estimator:
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',
                        random_state=1,
                        solver='lbfgs')
# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# fitting the logistic regression model on the reduced dataset:
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier= lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_)

#Assessing feature contributions

print("eigen_vecs shape:", eigen_vecs.shape)
print("eigen_values shape:", eigen_vals.shape)

loadings = eigen_vecs * np.sqrt(eigen_vals)

print("loadings shape: ", loadings.shape)

fig, ax = plt.subplots()
ax.bar( range(13), loadings[:, 0], align='center')
ax.set_ylabel("Loadings for PC1")
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1,1])
plt.show()

#Using scikit learn
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots()
ax.bar( range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel("Loadings for PC1")
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1,1])
plt.show()

#Stopped in page 154

#Supervised data compression via linear disciminant analysis (LDA)
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean( X_train_std[y_train == label], axis=0))
    print(f"MV {label}: {mean_vecs[label - 1]}\n")

#With the mean vectors, we calculate the within-class S_W
d = 13 #number of features
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X_train[y_train == label]:
        row, mv = row.reshape(d,1), mv.reshape(d,1)
        class_scatter += (row -mv).dot((row - mv).T)
    S_W += class_scatter

print('Within-class scatter matrix: 'f"{S_W.shape[0]}x{S_W.shape[1]}")
print('Class label distribution: ', np.bincount(y_train)[1:])

#Scaling within-class
d = 13 #number of features
S_W = np.zeros((d,d))
for label,mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

print('Scaled within-class scatter matrix: 'f"{S_W.shape[0]}x{S_W.shape[1]}")

#The scaled wihtin-class scatter matrix is just the same as the covariance matrix

#Calculating the between-class scatter matrix S_B
mean_overall = np.mean(X_train_std, axis=0)
print("shape of mean before reshape", mean_overall.shape)
mean_overall = mean_overall.reshape(d,1)

d = 13 #number of features
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i +1,:].shape[0]
    mean_vec = mean_vec.reshape(d, 1) #make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print("Between-class scatter matrix: " f"{S_B.shape[0]}x{S_B.shape[1]}")

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key= lambda k: k[0], reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, align='center',label='Individual discriminability')
plt.step(range(1,14), cum_discr, where='mid', label='Cumulative discriminability')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

w= np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

# X' = XW
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l,1]*(-1),
                c=c, label= f'class {l}', marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#Stopped at page 162
#LDA via scikit-learn  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier= lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc= 'lower left')
plt.tight_layout()
plt.show()

# Nonlinear dimensionality reduction and visualization
# Note: The development and application of nonlinear dimensionality reduction thechniques
# is also often referred to as manifold learning, where a manifold refers to a lower dimensional
# topologial space embedded in a high-dimensional space


# Visualizing data via t-distributed stochastic neighbor embedding

from sklearn.datasets import load_digits
digits = load_digits()

fig, ax = plt.subplots(1,4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()

print(digits.data.shape)
y_digits = digits.target
X_digits = digits.data

# from sklearn.manifold import TSNE
# tsne = TSNE(n_components= 2, init='pca', random_state=123)
# X_digits_tsne = tsne.fit_transform(X_digits)

# import matplotlib.patheffects as PathEffects
# def plot_projection(x, colors):

#     f = plt.figure(figsize=(8,8))
#     ax = plt.subplot(aspect='equal')
#     for i in range(10):
#         plt.scatter(x[colors == i, 0],
#                     x[colors == i, 1])
    
#     for i in range(10):
#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(i), fontsize= 24)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="W"),
#             PathEffects.Normal()
#         ])
#     plot_projection(X_digits_tsne, y_digits)
#     plt.show()
