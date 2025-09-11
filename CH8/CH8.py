import pyprind
import pandas as pd
import os
import sys
import numpy as np

## Need to do this only once
# import tarfile 
# with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
#     tar.extractall()

# basepath = 'aclImdb'

# labels = {'pos': 1, 'neg': 0}
# pbar = pyprind.ProgBar(50000, stream=sys.stdout)
# df = pd.DataFrame()

# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path = os.path.join(basepath, s, l)
#         for file in sorted(os.listdir(path)):
#             with open(os.path.join(path, file),
#                       'r', encoding='utf-8') as infile:
#                 txt = infile.read()

#             x = pd.DataFrame([[txt, labels[l]]], columns=['review', 'sentiment'])
#             df = pd.concat([df, x], ignore_index=False)

#             pbar.update()

# df.columns = ['review', 'sentiment']

# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('movie_data.csv', index=False, encoding='utf-8')


df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0":"review", "1": "sentiment"})
print(df.head(3))
print(df.shape)

#Introducing the bag-of-words model

#Transforming words into feature vectors

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array(["The sun is shining",
                 "The weater is sweet",
                 "The sun is shining, the weater is sweet,"
                 "and one and one is two"])
bag = count.fit_transform(docs)

print(count.vocabulary_)
print(bag.toarray())

#Assessing word relevancy via term frequency-inverse document frequency
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#Cleaning text data

print(df.loc[0, 'review'][-50:])

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-',''))
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))

print(preprocessor("</a>This :) is :( a test :-)!"))

# Processing documents into Tokens
def tokenizer(text):
    return text.split()
print(tokenizer('runners like running and thus they run'))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))

## Run this only once
# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot') 
 if w not in stop])

#Stopped at page 258

#Training a logistic regression model for document classification

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents = None,
                         lowercase= False,
                         preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range':[(1,1)],
        'vect__stop_words':[None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty':['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
]

# lr_tfidf = Pipeline([
#     ('vect', tfidf),
#     ('clf', LogisticRegression(solver='liblinear'))
# ])
# gs_lr_tdidf = GridSearchCV(lr_tfidf, small_param_grid,
#                            scoring='accuracy', cv=5,
#                            verbose=2, n_jobs=-1)
# gs_lr_tdidf.fit(X_train,y_train)

# #Stopped at beginning of page 259

# print(f"Best parameter set: {gs_lr_tdidf.best_params_}")

# print(f"CV Accuracy: {gs_lr_tdidf.best_score_:.3f}")

# clf= gs_lr_tdidf.best_estimator_
# print(f"Test Accuracy: {clf.score(X_test, y_test):.3f}")

import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

print(next(stream_docs(path='movie_data.csv')))

#Stopped at beginning of page 262