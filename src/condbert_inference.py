from helper.condbert import CondBert
from helper.embedding_similarity import EmbeddingSimilarity
from helper.bert_predictor import BertPredictor

import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pickle
import os
from tqdm.auto import tqdm, trange

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForMaskedLM.from_pretrained(model_name)
model.to(device)

# Loading the required input files for CondBERT

'''
Input files:

1) ../vocab/negative_words.txt
2) ../vocab/positive_words.txt
3) ../vocab/word2weight.pkl
4) ../vocab/token_toxicities.txt

'''

VOCAB_DIRNAME = "../vocab" 

with open(VOCAB_DIRNAME + "/negative_words.txt", "r") as f:
    s = f.readlines()
negative_words = list(map(lambda x: x[:-1], s))

with open(VOCAB_DIRNAME + "/positive_words.txt", "r") as f:
    s = f.readlines()
positive_words = list(map(lambda x: x[:-1], s))

with open(VOCAB_DIRNAME + '/word2weight.pkl', 'rb') as f:
    word2coef = pickle.load(f)

token_toxicities = []
with open(VOCAB_DIRNAME + '/token_toxicities.txt', 'r') as f:
    for line in f.readlines():
        token_toxicities.append(float(line))
token_toxicities = np.array(token_toxicities)
token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1))) 

# discourage meaningless tokens
for tok in ['.', ',', '-']:
    token_toxicities[tokenizer.encode(tok)][1] = 3

for tok in ['you']:
    token_toxicities[tokenizer.encode(tok)][1] = 0


# Initializing the CondBert  Model

def adjust_logits(logits, label=0):
    return logits - token_toxicities * 100 * (1 - 2 * label)

predictor = BertPredictor(model, tokenizer, max_len=250, device=device, label=0, contrast_penalty=0.0, logits_postprocessor=adjust_logits)

editor = CondBert(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities,
    predictor=predictor,
)

chooser = EmbeddingSimilarity(sim_coef=10, tokenizer=tokenizer)

# The inference part

'''
Here the input file we use is ../data/test_toxic.txt

the results are saved in the file ../results/condbert_results.txt

'''

input_file = "../data/test_toxic.txt"
output_file = "../results/condbert_results.txt"


with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for index,line in tqdm(enumerate(infile)):
        
        line = line.strip()

        # 97 percent length cutoff
        line = " ".join(line.split()[:80])


        try:
            transformed_line = editor.replacement_loop(line,
                                                        verbose=False, 
                                                        chooser=chooser,
                                                        n_tokens=(1, 2, 3), 
                                                        n_top=10
                                                    )
            
            outfile.write(transformed_line + '\n')
        except:
            outfile.write(line + '\n')

        



