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

#Should go back to page 254 (cleaning data) for a new review
#to understand more.