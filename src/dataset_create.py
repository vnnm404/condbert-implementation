import pandas as pd
import numpy as np
import torch
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import string
import random

random.seed(0)

gpu = 'cuda:0' if cuda.is_available() else 'cpu'

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

df = pd.read_csv("./Jigsaw_2018_test.csv", usecols = ['comment_text'])

def clean_text(text):
    text = text.replace('\n', ' ')  # replace newline characters with space
    text = text.lower()  # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    
    return text

all_sentences = []
print("Cleaning Sentences")
for index, row in df.iterrows():
    text_content = row['comment_text']

    sentences = sent_tokenize(text_content)

    cleaned_sentences = [clean_text(sentence) for sentence in sentences]
    all_sentences.extend(cleaned_sentences)

class SentData(Dataset):
    def __init__(self, tokenizer, max_len, sentences):
        self.tokenizer = tokenizer
        self.text = sentences
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'text': self.text[index],
        }

toxic_sent = []
normal_sent = []
toxic_val = []

sent_ds = SentData(tokenizer, MAX_LEN, all_sentences)
params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
loader = DataLoader(sent_ds, **params)

model = torch.load("./roberta_clf.bin")
model.eval()



def test(epoch, model_1):
    model_1.eval()
    for _,data in tqdm(enumerate(loader, 0)):
        ids = data['ids'].to(gpu, dtype = torch.long)
        mask = data['mask'].to(gpu, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(gpu, dtype = torch.long)
        text = data['text']

        outputs = model_1(ids, mask, token_type_ids).logits
        big_val, big_idx = torch.max(outputs.data, dim=1)



        for i in range(len(outputs)):
          if big_idx[i] == 0:
            normal_sent.append(text[i])
          else:
            toxic_sent.append(text[i])
            toxic_val.append(big_val[i])

print("Test has started")
test(1,model)
print("Test has ended")
sample_length = min(len(toxic_sent), 10000)
print(f"Sample Length is {sample_length}")

indexed_list = [(index, value) for index, value in enumerate(toxic_val)]
indexed_list.sort(key=lambda x: x[1])

sorted_toxic_val = [value for index, value in indexed_list]
sorted_toxic_sent = [toxic_sent[index] for index, _ in indexed_list]

with open("/home2/vnnm/anlp/project/roberta_finetune/test_toxic.txt", "w") as f:
  for sent in sorted_toxic_sent[:sample_length]:
    f.write(f"{sent}\n")

with open("/home2/vnnm/anlp/project/roberta_finetune/test_normal.txt", "w") as f:
  for sent in random.sample(normal_sent, sample_length):
    f.write(f"{sent}\n")




