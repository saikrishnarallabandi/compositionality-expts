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

print_flag = 0
logfile_name = 'log_modelpredictions_questionsonly'
hk = open(logfile_name,'w')
hk.close()


def sample_gumbel(shape, eps=1e-10, out=None):
   """
   Sample from Gumbel(0, 1)
   based on
   https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
   (MIT license)
   """
   U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
   return - torch.log(eps - torch.log(U + eps))

def gumbel_argmax(logits, dim):
   # Draw from a multinomial distribution efficiently
   #print("Shape of gumbel input: ", logits.shape)
   return logits + sample_gumbel(logits.size(), out=logits.data.new())
   return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


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

trainquestion_ints = []
trainanswer_ints = []
trainimage_ids = []
trainfeatures = []
h = open('answers_vqa2014_jacobandreassplit.txt','w')
for t in train_dict:
 try:
   if len(t.keys()) > 0:
    question = t['question_str']
    answer = t['valid_answers'][0]
    image_id = t['image_name'] + '.jpg'
    h.write(answer + '\n')
    l = []
    for q in question.split():
        k_l = questions_wids[q]
        l.append(k_l)
    trainquestion_ints.append(l)
    trainanswer_ints.append(answer_wids[answer])
    trainimage_ids.append(image_id)
    trainfeatures.append(imageid2features[image_id])
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

unk_id = answer_wids['<unk>']
valquestion_ints = []
valanswer_ints = []
valimage_ids = []
valfeatures = []
for v in val_dict:
 try:
   if len(v.keys()) > 0:
    question = v['question_str']
    answer = v['valid_answers'][0]
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



# Load image features
import pickle
with open('imageid2features.pkl', 'rb') as f:
   imageid2features = pickle.load(f)

print("Succesfully loaded image id to features")

class jointvqacaptions_dataset(Dataset):

      def __init__(self, questions, features, answers):
         self.questions = questions
         self.features = features
         self.answers = answers

      def __len__(self):
        return len(self.features)

      def __getitem__(self, item):
        return self.questions[item], self.features[item], self.answers[item]

      def collate_fn(self, batch):

       question_lengths = [len(x[0]) for x in batch]
       max_question_len = np.max(question_lengths) + 1
       a =  np.array([ self._pad(x[0], max_question_len)  for x in batch ], dtype=np.int)

       b = [x[1] for x in batch]
       c =  np.array([ x[2]  for x in batch ], dtype=np.int)

       a = torch.LongTensor(a)
       b = torch.FloatTensor(b)
       c = torch.LongTensor(c)

       return a,b,c

      def _pad(self, seq, max_len):
        return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


jvcdset = jointvqacaptions_dataset(trainquestion_ints, trainfeatures, trainanswer_ints)
train_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )
num_questionclasses = len(questions_wids)
num_answerclasses = len(answer_wids)




class baseline_model(nn.Module):

   def __init__(self, question_embed_dim, caption_embed_dim, answer_embed_dim, num_classes, num_answerclasses):
      super(baseline_model, self).__init__()
      hidden_dim = 128
      resnetfeats_dim = 2048
      self.question_embedding = nn.Embedding(num_classes, question_embed_dim)
      self.caption_embedding = nn.Embedding(num_classes, caption_embed_dim)
      self.embed2lstm = nn.Linear(question_embed_dim,question_embed_dim*2)
      self.lstm = nn.LSTM(question_embed_dim*4, hidden_dim, 2, batch_first = True, bidirectional=True)
      self.hidden2out =  nn.Linear(hidden_dim*2, num_answerclasses)
      self.dropout = nn.Dropout(0.3)
      self.image_embedding = nn.Linear(resnetfeats_dim,question_embed_dim*2)

   def forward(self,question, features):
      question_embedding = self.question_embedding(question)
      question_embedding = F.relu(self.embed2lstm(question_embedding))
      question_embedding = self.dropout(question_embedding) 

      if print_flag:
         print("Shape of question embedding: ", question_embedding.shape)
         print("Shape of features: ", features.shape)

      # Repeat image feature embedding and concatenate
      features = F.relu(self.image_embedding(features))
      features.unsqueeze_(1)
      length_desired = question.shape[1]
      features = features.repeat(1, length_desired, 1) 

      if print_flag:
         print("Shape of question embedding: ", question_embedding.shape)
         print("Shape of features: ", features.shape)

      total_embedding = torch.cat([question_embedding, features], dim=-1)     
      if print_flag:
         print("Shape of total embedding: ", total_embedding.shape)

      x, hidden = self.lstm(total_embedding)

      if print_flag:
         print("Shape of lstm output: ", x.shape)
      return self.hidden2out(x)[:,-1]


question_embed_dim = 128
caption_embed_dim = 128
answer_embed_dim = 128

model = baseline_model(question_embed_dim, caption_embed_dim, answer_embed_dim, num_questionclasses, num_answerclasses)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')




def train():
  model.train()
  curr_step = 0
  total_loss = 0
  for i, data in enumerate(train_loader):
    question, features, answer = data[0], data[1], data[2]
    question, features, answer = Variable(question), Variable(features), Variable(answer)
    if print_flag:
        print("Shape of question, caption and answer is ", question.shape,  answer.shape)      
    if torch.cuda.is_available():
          question, features, answer = question.cuda(), features.cuda(), answer.cuda()

    answer_predicted = model(question, features)
    if print_flag:
        print("Shape of predicted answer is ", answer_predicted.shape, " and that of original answer was ", answer.shape)        
    loss = criterion(answer_predicted.view(-1, num_answerclasses), answer.view(-1))
    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

    if i % 1000 == 1:
       #val_loss = val()
       print(" Train Loss after ", i , " updates: ", total_loss/(i+1))
      
  return total_loss/(i+1)

for epoch in range(10):
   train_loss = train()
   val_loss = val()
   print("After epoch ", epoch, " train loss: ", train_loss, " val loss: ", val_loss)
