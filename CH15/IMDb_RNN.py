import torch
from torchtext.datasets import IMDB
from datasets import load_dataset
from torch.utils.data.dataset import random_split
import re
from collections import Counter, OrderedDict

ds = load_dataset('imdb')

train_dataset = list(zip(ds['train']['label'], ds['train']['text']))
test_dataset = list(zip(ds['test']['label'], ds['test']['text']))

## Step 1: create the datasets
torch.manual_seed(1)
train_dataset, valid_dataset = random_split(
    list(train_dataset), [20000, 5000])


## Step 2: find unique tokens (words)

token_counts = Counter()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized


for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
 
    
print('Vocab-size:', len(token_counts))
