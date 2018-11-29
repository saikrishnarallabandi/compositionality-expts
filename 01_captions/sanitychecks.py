import numpy as np
import os
from torch.utils.data import Dataset
from collections import defaultdict
import sys
from data_loader_barebones import vqa_dataset
from nltk import word_tokenize
from torch.utils.data import DataLoader
import json
import torch.nn as nn
import torch
from torch.autograd import Variable
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

print_flag = 1
logfile_name = 'log_sanitychecks'
hk = open(logfile_name,'w')
hk.close()


# Load the imdb files
train_dict = np.load('imdb_train2014.npy')
val_dict = np.load('imdb_val2014.npy')

questions_wids = defaultdict(lambda: len(questions_wids))
questions_wids['_PAD'] = 0
questions_wids['UNK'] = 1
answer_wids = defaultdict(lambda: len(answer_wids))

# Load image features
import pickle
with open('imageid2features.pkl', 'rb') as f:
   imageid2features = pickle.load(f)
#imageid2features.pkl
with open('imageid2captions.pkl', 'rb') as f:
   imageid2captions = pickle.load(f)

trainquestion_ints = []
trainanswer_ints = []
trainimage_ids = []
trainfeatures = []
traincaption_ints = []
trainquestion_ids = []
h = open('answers_vqa2014_jacobandreassplit_image_caption.txt','w')
for t in train_dict:
 try:
   if len(t.keys()) > 0:
    question = t['question_str']
    answer = t['valid_answers'][0]
    image_id = t['image_name'] + '.jpg'
    question_id = t['question_id']
    h.write(answer + '\n')
    l = []
    for q in question.split():
        k_l = questions_wids[q]
        l.append(k_l)
    trainquestion_ints.append(l)
    trainanswer_ints.append(answer_wids[answer])
    trainimage_ids.append(image_id)
    trainfeatures.append(imageid2features[image_id])
    trainquestion_ids.append(question_id)
    l = []
    caption = imageid2captions[image_id]
    for c in caption.split():
        k_l = questions_wids[c]
        l.append(k_l)
    traincaption_ints.append(l)

    #print("Shape of image id to features is ", imageid2features[image_id].shape)

 except AttributeError:
    print("This seems weird: ", t)
h.close()
print("There are these many items: ", len(trainquestion_ints))

assert len(trainquestion_ints) == len(trainanswer_ints)

question_i2w =  {i:w for w,i in questions_wids.items()}
answer_i2w = {i:w for w,i in answer_wids.items()}

print("The number of question classes: ", len(question_i2w.keys()))
print("The number of answer classes: ", len(answer_i2w.keys()))

with open('imageid2features_val.pkl', 'rb') as f:
   imageid2features_val = pickle.load(f)
# imageid2features.pkl
with open('imageid2captions_val.pkl', 'rb') as f:
   imageid2captions_val = pickle.load(f)

unk_id = answer_wids['<unk>']
valquestion_ints = []
valanswer_ints = []
valimage_ids = []
valfeatures = []
valcaption_ints = []
valquestion_ids = []
for v in val_dict:
 try:
   if len(v.keys()) > 0:
    question = v['question_str']
    answer = v['valid_answers'][0]
    image_id = v['image_name'] + '.jpg'
    question_id = v['question_id']

    l = []
    for q in question.split():
        if q in questions_wids: 
            k_l = questions_wids[q]
            l.append(k_l)
        else:
            l.append(1) 
    valquestion_ints.append(l)
    if answer in answer_wids:
       k_l = answer_wids[answer]
    else:
       k_l = unk_id
    valanswer_ints.append(k_l)
    valimage_ids.append(image_id)
    valfeatures.append(imageid2features_val[image_id])
    caption = imageid2captions_val[image_id]
    valquestion_ids.append(question_id)
    l = []
    for c in caption.split():
      if c in questions_wids:
         k_l = questions_wids[c]
         l.append(k_l)
      else:
         l.append(1)
    valcaption_ints.append(l)

    #image_id = v['image_id']
    #valfeatures.append(
 except AttributeError:
    print("This seems weird: ", t)
h.close()
print("There are these many items: ", len(valquestion_ints))

question_i2w =  {i:w for w,i in questions_wids.items()}
answer_i2w = {i:w for w,i in answer_wids.items()}

print("The number of question classes: ", len(question_i2w.keys()))
print("The number of answer classes: ", len(answer_i2w.keys()))



print("Succesfully loaded image id to features")
print("Exiting")

class jointvqacaptions_dataset(Dataset):

      def __init__(self, questions, features, answers, captions, question_ids):
         self.questions = questions
         self.features = features
         self.answers = answers
         self.captions = captions
         self.question_ids = question_ids

      def __len__(self):
        return len(self.features)

      def __getitem__(self, item):
        return self.questions[item], self.features[item], self.answers[item], self.captions[item], self.question_ids[item]

      def collate_fn(self, batch):

       question_lengths = [len(x[0]) for x in batch]
       caption_lengths = [len(x[3]) for x in batch]
       question_lengths += caption_lengths
       max_question_len = np.max(question_lengths) + 1

       a =  np.array([ self._pad(x[0], max_question_len)  for x in batch ], dtype=np.int)
       b = [x[1] for x in batch]
       c =  np.array([ x[2]  for x in batch ], dtype=np.int)
       d =  np.array([ self._pad(x[3], max_question_len)  for x in batch ], dtype=np.int)       
       e = np.array([ x[4]  for x in batch ], dtype=np.int)

       a = torch.LongTensor(a)
       b = torch.FloatTensor(b)
       c = torch.LongTensor(c)
       d = torch.LongTensor(d)
       e = torch.LongTensor(e)

       return a,b,c,d,e

      def _pad(self, seq, max_len):
        return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


jvcdset = jointvqacaptions_dataset(trainquestion_ints, trainfeatures, trainanswer_ints, traincaption_ints, trainquestion_ids)
train_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )
num_questionclasses = len(questions_wids)
num_answerclasses = len(answer_wids)


jvcdset = jointvqacaptions_dataset(valquestion_ints, valfeatures, valanswer_ints, valcaption_ints, trainquestion_ids)
val_loader = DataLoader(jvcdset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )



question_embed_dim = 128
caption_embed_dim = 128
answer_embed_dim = 128


def train():
  curr_step = 0
  total_loss = 0
  for i, data in enumerate(val_loader):
    question, features, answer, caption, qid  = data[0], data[1], data[2], data[3], data[4]
    question, features, answer, caption, qid = Variable(question), Variable(features), Variable(answer), Variable(caption), Variable(qid)
    
    if i % 1000 == 1:
     groundtruth_question = ' '.join(question_i2w[k] for k in question.numpy()[0])
     groundtruth_caption = ' '.join(question_i2w[k] for k in caption.numpy()[0])

     hk = open(logfile_name,'a')
     hk.write("  The ground truth question during test was " +  groundtruth_question + '\n')
     hk.write("  The ground truth caption during test was " +  groundtruth_caption + '\n')
     groundtruth_answer = answer_i2w[answer.numpy()[0]]
     groundtruth_qid = qid.numpy()[0]
     hk.write("  The ground truth answer during test was " +  str(groundtruth_answer) + '\n')
     hk.write("  The ground truth question id during test was " +  str(groundtruth_qid) + '\n')
     hk.write('\n')
     hk.close()


for epoch in range(1):
   train_loss = train()
