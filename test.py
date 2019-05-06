from voc_cloth.voc import vocab_api
from voc_cloth.CLOTHapi import CLOTH_api
import data_util
import json
import csv
from voc_cloth.data_util import ClothSample
from FitSentence.FitSentenceAPI import fit_the_best_sentence_api
import sys
from reading.RCapi import reading_comprehension_api
import docx

def doc2json(file_name):
    doc = docx.Document(file_name)
    for para in doc.paragraphs:
        print(para.text)

doc2json(u'test.docx')
#voc
questions = []
text = "Maria didn’t want to deliver the bad news to David about his failing the job interview. She herself was quite upset about it."
questions.append({'sentence':text, 'answer':'upset'})
text = "The newcomer speaks with a strong Irish accent; he must be from Ireland."
questions.append({'sentence':text, 'answer':'accent'})

file_list = data_util.get_json_file_list('/home/dsj/examkeeper/web/voc_cloth/CLOTH/test')
all_data = []
for file_name in file_list:
    data = json.loads(open(file_name, 'r').read())
    data['high'] = 0
    if 'high' in file_name:
        data['high'] = 1
    all_data.append(data)

all_data = all_data[:1]
print(len(all_data))
print(all_data[0])

#output_prob = CLOTH_api(all_data)
#print(output_prob)


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


#print(fit_the_best_sentence_api(questions[:20]))


# In[ ]:

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


#output_probability = reading_comprehension_api(questions)
#print(output_probability)


