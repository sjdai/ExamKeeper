#!/usr/bin/env python
# coding: utf-8

# # Stuff you need to import in the first place
# - install pytorch-bert from https://github.com/huggingface/pytorch-pretrained-BERT

# In[1]:


"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import random
import data_util
from data_util import ClothSample
import numpy as np
import torch
import time
from voc_cloth.pytorch_pretrained_bert.modeling import BertForCloth
from voc_cloth.pytorch_pretrained_bert.tokenization import BertTokenizer
from voc_cloth.pytorch_pretrained_bert.optimization import BertAdam
from voc_cloth.pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools
import json


# In[2]:


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForCloth.from_pretrained('bert-large-uncased',
          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
model.to(device).eval()


# # API for calling when questions are put into Exam Keeper

# In[5]:


import time

BATCH_SIZE=1 # don't change!
CACHE_SIZE=256

def CLOTH_api(questions):#, num_candidates):

    data = data_util.Preprocessor('bert-large-uncased', tokenizer, all_data).data_objs

    valid_data = data_util.Loader(data, CACHE_SIZE, BATCH_SIZE, device)

    # output_probability = torch.zeros(0, num_candidates).cuda()
    output_probability = []

    for inp, tgt in valid_data.data_iter(shuffle=False):
        with torch.no_grad():
            out, bsz, opnum, num_candidates = model(inp, tgt)
            out = out.view(-1, num_candidates)

            # output_probability = torch.cat((output_probability, torch.softmax(out, 1)), 0)
            output_probability.append(torch.softmax(out, 1))
    return output_probability


# # Example

# In[6]:


file_list = data_util.get_json_file_list('/home/dsj/examkeeper/web/voc_cloth/CLOTH/test')
all_data = []
#max_article_len = 0
for file_name in file_list:
    data = json.loads(open(file_name, 'r').read())
    data['high'] = 0
    if ('high' in file_name):
        data['high'] = 1
    all_data.append(data)


# In[6]:


# An example of reading comprehension input
all_data = all_data[:1]
print(len(all_data))
print(all_data[0])
# when used for exam keeper, add 'high': 0 for all samples

print(all_data)
# In[7]:


output_probability = CLOTH_api(all_data)


# In[8]:


print(output_probability)


# In[ ]:




