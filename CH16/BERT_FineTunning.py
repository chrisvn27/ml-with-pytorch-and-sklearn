import gzip
import shutil
import time

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext

import transformers
from transformers import DistilBertTokenizerFast 
from transformers import DistilBertForSequenceClassification

from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 3

url = "https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz"
filename = url.split("/")[-1]

with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with gzip.open('movie_data.csv.gz', 'rb') as f_in:
    with open('movie_data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

df = pd.read_csv('movie_data.csv')
print(df.head(3))

train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values

valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values

test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

#Tokenizing the dataset
tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-uncased'
)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset =  IMDbDataset(valid_encodings, valid_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased'
)
model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            ### Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
        
        return correct_pred.float()/num_examples * 100
    
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    
    model.train()

    for batch_idx, batch in enumerate(train_loader):

        ### Prepare data
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        ### Forward pass
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss, logits = outputs['loss'], outputs['logits']

        ### Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        ### Logging
        if not batch_idx % 250:
            print(f"Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d}",
                  f" | Batch {batch_idx:04d}/{len(train_loader):04d}",
                  f" | Loss: {loss:.4f}")
            
    model.eval()

    with torch.no_grad():
        print(f"Training accuracy: {compute_accuracy(model, train_loader, DEVICE):.2f}%",
              f"\nValid accuracy: {compute_accuracy(model, valid_loader, DEVICE):.2f}%")
    
    print(f"time elapsed: {(time.time() - start_time)/60:.2f} min")

print(f"Total Training Time: {(time.time() - start_time)/60:.2f} min")
print(f"Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%")

# Fine-tunning a transformer more conveniently using the 
# Trainer API
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.train();

optim = torch.optim.Adam(model.parameters(), lr=5e-5)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optim, None) #optim and learning rate scheduler
)

from datasets import load_metric
import numpy as np

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #note: logits are a numpy array, not a pytorch tensor
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optim, None) #optim and learning rate scheduler
)

start_time = time.time()
trainer.train()

print(trainer.evaluate())

model.eval()
model.to(DEVICE)

print(f"Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%")
