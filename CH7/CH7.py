from scipy.special import comb
import math

def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [comb(n_classifier, k) *
             error **k *
             (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

print(ensemble_error(n_classifier=11, error=0.25))

import numpy as np
import matplotlib.pyplot as plt

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error = error) for error in error_range]

plt.plot(error_range, ens_errors, label='Ensemble error', linewidth = 2)
plt.plot(error_range, error_range, label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()

# Implementing a simple majority vote classifier
print(np.argmax(np.bincount([0,0,1], weights = [0.2, 0.2, 0.6])))

ex = np.array([[0.9,0.1],
               [0.8, 0.2],
               [0.4, 0.5]])

p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print(p)
print(np.argmax(p))


# Implementing MajorityVoteClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator

# class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, classifiers, vote='classlabel', weights=None):
        
#         self.classifiers = classifiers
#         self.named_classifiers ={
#             key: value for key, value in _name_estimators(classifiers)
#         }
#         self.vote = vote
#         self.weights= weights

#     def fit(self, X, y):
#         if self.vote not in ('probability', 'classlabel'):
#             raise ValueError(f"vote must be 'probability' "
#                              f" or 'classlabel'"
#                              f"; got (vote={self.vote})")
#         if self.weights and len(self.weights) != len(self.classifiers):
#             raise ValueError(f"Number of classifiers and"
#                              f" weights must be equal"
#                              f"; got {len(self.weights)} weights"
#                              f" {len(self.classifiers)} classifiers")
#         # Use LabelEncoder to ensure class labels start
#         # with 0, which is important for np.argmax
#         # class in self.predict
#         self.lablenc_ = LabelEncoder()
#         self.lablenc_.fit(y)
#         self.classes_ = self.lablenc_.classes_
#         self.classifiers_ = []
        
#         for clf in self.classifiers:
#             fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
#             self.classifiers_.append(fitted_clf)
#         return self
    
#     def predict(self, X):
#         if self.vote == 'probability':
#             maj_vote = np.argmax(self.predict_proba(X), axis=1)
#         else: # 'class label vote'
#             predictions = np.asarray([
#                 clf.predict(X) for clf in self.classifiers_
#             ]).T
        
#         maj_vote = np.apply_along_axis(lambda x: np.argmax(
#             np.bincount(x, weights= self.weights)
#         ),
#         axis=1, arr=predictions
#         )
#         maj_vote = self.lablenc_.inverse_transform(maj_vote)
#         return maj_vote
    
#     def predict_proba(self, X):
#         probas = np.asarray([clf.predict_proba(X)
#                              for clf in self.classifiers_])
#         avg_proba = np.average(probas, axis=0, weights= self.weights)
#         return avg_proba
#     def get_params(self, deep = True):
#         if not deep:
#             return super().get_params(deep=False)
#         else:
#             out = self.named_classifiers.copy()
#             for name, step in self.named_classifiers.items():
#                 for key,value in step.get_params(
#                     deep=True).items():
#                     out[f'{name}__{key}'] = value
#             return out


# Scikit-learn 0.16 and newer requires reversing the parent classes
# See https://github.com/rasbt/machine-learning-book/discussions/205 for more details
import sklearn
base_classes = (ClassifierMixin, BaseEstimator) if sklearn.__version__ >= "0.16" else (BaseEstimator, ClassifierMixin)

class MajorityVoteClassifier(*base_classes):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='classlabel')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        y : array-like, shape = [n_examples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' or 'classlabel'"
                             f"; got (vote={self.vote})")

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and weights must be equal'
                             f'; got {len(self.weights)} weights,'
                             f' {len(self.classifiers)} classifiers')

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_examples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_examples, n_classes]
            Weighted average probability for each class per example.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out



# Using the majority voting principle to make predictions

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1,2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test =\
    train_test_split(X,y, test_size=0.5,random_state=1,stratify=y)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='l2',
                          C=0.001,
                          solver='lbfgs',
                          random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc',StandardScaler()],
                  ['clf', clf3]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    
    print(f"ROC AUC: {scores.mean():.2f} "
          f"(+/- {scores.std():.2f}) [{label}]")
    
#Stopped at page 216
mv_clf = MajorityVoteClassifier(
    classifiers=[pipe1,clf2,pipe3]
)
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f"ROC AUC: {scores.mean():.2f} "
          f"(+/- {scores.std():.2f}) [{label}]")
    