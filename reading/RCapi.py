#!/usr/bin/env python
# coding: utf-8

# # Stuff you need to import in the first place
# - install pytorch-bert from https://github.com/huggingface/pytorch-pretrained-BERT

# In[1]:


from reading.run_race import RaceExample, convert_examples_to_features, select_field, accuracy
import os
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertModel

##############
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools


# In[2]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model_state_dict = torch.load("/home/ting/BERT-RACE/base_models/pytorch_model.bin")
model = BertForMultipleChoice.from_pretrained('bert-base-uncased',
    # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
    state_dict=model_state_dict,
    num_choices=4)


# In[3]:


model.cuda().eval()


# # API for calling when questions are put into Exam Keeper

# In[4]:


import time

BATCH_SIZE=8

def feedexample(question):
    example = RaceExample(
        race_id = question['id'],
        context_sentence = question['article'],
        start_ending = question['question'],

        ending_0 = question['options'][0],
        ending_1 = question['options'][1],
        ending_2 = question['options'][2],
        ending_3 = question['options'][3],
        label = question['truth'])
    return example

def reading_comprehension_api(questions):

    eval_examples = [feedexample(question) for question in questions]

    eval_features = convert_examples_to_features(eval_examples, tokenizer, 512, True)

    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)

    output_probability = torch.zeros((len(eval_features), 4))

    for step, batch in enumerate(eval_dataloader):
        start = time.time()
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu()
        end = time.time()
        # print(end - start)

        output_probability[step*BATCH_SIZE:(step+1)*BATCH_SIZE] = torch.softmax(logits, 1)
    return output_probability


# # Example

# In[5]:


questions = []
filenames = ['21130.txt', '2358.txt', '4135.txt', '20965.txt', '21130.txt', '2358.txt', '4135.txt','21130.txt', '2358.txt', '4135.txt']
for filename in filenames:
    with open('/home/ting/BERT-RACE/RACE/test/high/' + filename, 'r', encoding='utf-8') as fpr:
        data_raw = json.load(fpr)
        article = data_raw['article']
        for i in range(len(data_raw['answers'])):
            truth = ord(data_raw['answers'][i]) - ord('A')
            question = data_raw['questions'][i]
            options = data_raw['options'][i]

            questions.append(
                {'id':filename+'-'+str(i),
                 'article':article,
                 'question':question,
                 'options':options,
                 'truth':truth
                }
            )


# In[6]:


# An example of reading comprehension input
print(len(questions))
print(questions[0])


# In[7]:


output_probability = reading_comprehension_api(questions)
print(output_probability)


# In[ ]:





# In[ ]:




