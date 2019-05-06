#!/usr/bin/env python
# coding: utf-8

# # Stuff you need to import in the first place
# - install pytorch-bert from https://github.com/huggingface/pytorch-pretrained-BERT

# In[1]:


from FitSentence.run_swag import SwagExample, convert_examples_to_features, select_field, accuracy
import csv
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer


# In[2]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model_state_dict = torch.load("/home/ting/pytorch-pretrained-BERT/base_models/weights-bert-base-uncased")
model = BertForMultipleChoice.from_pretrained('bert-base-uncased',
    state_dict=model_state_dict,
    num_choices=4)
model.cuda().eval()


# # API for calling when questions are put into Exam Keeper

# In[3]:


def feedexample(question):
    example = SwagExample(
        swag_id = question['id'],
        context_sentence = question['context_sentence'],
        start_ending = question['start_ending'],

        ending_0 = question['ending_0'],
        ending_1 = question['ending_1'],
        ending_2 = question['ending_2'],
        ending_3 = question['ending_3'],

        label = question['label']
        )
    return example

def fit_the_best_sentence_api(questions):

    eval_examples = [feedexample(question) for question in questions]

    eval_features = convert_examples_to_features(eval_examples, tokenizer, 512, True)

    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8)

    output_probability = torch.zeros((len(eval_features), 4))

    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu()

        output_probability[step*8:(step+1)*8] = torch.softmax(logits, 1)
    return output_probability


# # Example

# In[4]:


with open('/home/ting/swagaf/data/train.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = []
    for line in reader:
        if sys.version_info[0] == 2:
            line = list(unicode(cell, 'utf-8') for cell in line)
        lines.append(line)


# In[5]:


questions = []
for line in lines[1:]:

    questions.append(
        {'id':line[2],
         'context_sentence':line[4],
         'start_ending':line[5],
         'ending_0': line[7],
         'ending_1': line[8],
         'ending_2': line[9],
         'ending_3': line[10],
         'label': int(line[11])
        }
    )


# In[6]:


# An example of fit the best sentence input
'''
When we take input from teachers, start_ending might be "", which should be ok.
Context_sentence is the first sentence and ending_? is the second sentences.
We should pick the best second sentence as the continuation for the first sentence.
'''
print(questions[1])


# In[7]:


print(fit_the_best_sentence_api(questions[:20]))


# In[ ]:




