import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

#ft
from run_swag import SwagExample, convert_examples_to_features, select_field, accuracy
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

#rc
from run_race import RaceExample, convert_examples_to_features, select_field, accuracy
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

#cloth
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
from pytorch_pretrained_bert.modeling import BertForCloth
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools
import json




# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval().cuda()

class voc():
    def vocab_api(questions):
        all_candidates = []
        all_candidate_probabilities = []

        for question in questions:
            tokenized_text = tokenizer.tokenize(question['sentence'])

            # search for masked_index
            for idx, token in enumerate(tokenized_text):
                if token == question['answer']:
                    masked_index = idx
                    tokenized_text[masked_index] = '[MASK]'
                    break

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids = [0] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            segments_tensors = torch.tensor([segments_ids]).cuda()

            predictions = model(tokens_tensor, segments_tensors)

            predicted_probabilities, predicted_indexes = torch.topk(predictions[0, masked_index], 1000)

            candidate_probabilities = []
            candidates = []
            for idx, predicted_index in enumerate(predicted_indexes):
                index = predicted_index.item()
                word = tokenizer.convert_ids_to_tokens([index])[0]
                if word[0] == question['answer'][0] and word[-1] == question['answer'][-1]:
                    print(word, predicted_probabilities[idx])
                    candidate_probabilities.append(predicted_probabilities[idx].item())
                    candidates.append(word)
                    if len(candidates) >= 4:

                        break
            candidate_probabilities = torch.softmax(torch.tensor(candidate_probabilities), 0)

            all_candidates.append(candidates)
            all_candidate_probabilities.append(candidate_probabilities)
        return (all_candidates, all_candidate_probabilities)

class FitBestSentence():
    #FitSentence
    from run_swag import SwagExample, convert_examples_to_features, select_field, accuracy
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

class ReadingComprehension():
    #Reading Comprehension
    from run_race import RaceExample, convert_examples_to_features, select_field, accuracy
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

class Cloth():
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
    from pytorch_pretrained_bert.modeling import BertForCloth
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    from pytorch_pretrained_bert.optimization import BertAdam
    from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    import functools
    import json


    # In[2]:


    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForCloth.from_pretrained('bert-large-uncased',
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))


    # In[4]:


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:1')
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

