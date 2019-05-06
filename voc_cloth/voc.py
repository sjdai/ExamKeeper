#!/usr/bin/env python
# coding: utf-8

# In[89]:


import torch
from voc_cloth.pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval().cuda()


# In[90]:
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


# In[91]:
'''
if __name__ == "__main__":
    questions = []
    text = "Maria didn’t want to deliver the bad news to David about his failing the job interview. She herself was quite upset about it."
    questions.append({'sentence':text, 'answer':'upset'})
    text = "The newcomer speaks with a strong Irish accent; he must be from Ireland."
    questions.append({'sentence':text, 'answer':'accent'})
    voc = voc()




    all_candidates, all_candidate_probabilities = voc.vocab_api(questions)


    print(all_candidates)
    print(all_candidate_probabilities)


# In[ ]:



'''
