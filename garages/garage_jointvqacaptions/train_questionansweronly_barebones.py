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

trainquestion_ints = []
trainanswer_ints = []
h = open('answers_vqa2014_jacobandreassplit.txt','w')
for t in train_dict:
 try:
   if len(t.keys()) > 0:
    question = t['question_str']
    answer = t['valid_answers'][0]
    h.write(answer + '\n')
    l = []
    for q in question.split():
        k_l = questions_wids[q]
        l.append(k_l)
    trainquestion_ints.append(l)
    trainanswer_ints.append(answer_wids[answer])
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
 except AttributeError:
    print("This seems weird: ", t)
h.close()
print("There are these many items: ", len(valquestion_ints))

question_i2w =  {i:w for w,i in questions_wids.items()}
answer_i2w = {i:w for w,i in answer_wids.items()}

print("The number of question classes: ", len(question_i2w.keys()))
print("The number of answer classes: ", len(answer_i2w.keys()))

#sys.exit()


class jointvqacaptions_dataset(Dataset):

      def __init__(self, questions, captions, answers):
         self.questions = questions
         self.captions = captions
         self.answers = answers

      def __len__(self):
        return len(self.captions)

      def __getitem__(self, item):
        return self.questions[item], self.captions[item], self.answers[item]

      def collate_fn(self, batch):

       question_lengths = [len(x[0]) for x in batch]
       max_question_len = np.max(question_lengths) + 1
       a =  np.array([ self._pad(x[0], max_question_len)  for x in batch ], dtype=np.int)

       caption_lengths = [len(x[1]) for x in batch]
       max_caption_len = np.max(caption_lengths) + 1
       b =  np.array([ self._pad(x[1], max_caption_len)  for x in batch ], dtype=np.int)

       c =  np.array([ x[2]  for x in batch ], dtype=np.int)

       a = torch.LongTensor(a)
       b = torch.LongTensor(b)
       c = torch.LongTensor(c)

       return a,b,c

      def _pad(self, seq, max_len):
        return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


jvcdset = jointvqacaptions_dataset(trainquestion_ints, trainquestion_ints, trainanswer_ints)
train_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )
num_questionclasses = len(questions_wids)
num_answerclasses = len(answer_wids)

jvcdset = jointvqacaptions_dataset(valquestion_ints, valquestion_ints, valanswer_ints)
val_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )



class baseline_model(nn.Module):

   def __init__(self, question_embed_dim, caption_embed_dim, answer_embed_dim, num_classes, num_answerclasses):
      super(baseline_model, self).__init__()
      hidden_dim =128
      self.question_embedding = nn.Embedding(num_classes, question_embed_dim)
      self.caption_embedding = nn.Embedding(num_classes, caption_embed_dim)
      self.embed2lstm = nn.Linear(question_embed_dim,question_embed_dim*2)
      self.lstm = nn.LSTM(question_embed_dim*2, hidden_dim, 2, batch_first = True, bidirectional=True)
      self.hidden2out =  nn.Linear(hidden_dim*2, num_answerclasses)
      self.dropout = nn.Dropout(0.3)

   def forward(self,question):
      question_embedding = self.question_embedding(question)
      question_embedding = F.relu(self.embed2lstm(question_embedding))
      question_embedding = self.dropout(question_embedding) 

      if print_flag:
         print("Shape of question embedding: ", question_embedding.shape)

      x, hidden = self.lstm(question_embedding)
      
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


def test():
  model.eval()
  y_predicted = []
  for i, data in enumerate(val_loader):
    question, answer = data[0], data[2]
    if print_flag:
        print("Shape of question, caption and answer in test is ", question.shape,  answer.shape, question, answer[0])      
    question, answer = Variable(question[0,:]), Variable(answer[0])
    question1 = question
    groundtruth_question = ' '.join(question_i2w[k] for k in question1.detach().numpy())
    hk = open(logfile_name,'a')
    hk.write("  The ground truth question during test was " +  groundtruth_question + '\n')
    #print("The ground truth question during test was ", groundtruth_question ) 
    answer1 = answer.item()
    groundtruth_answer = answer_i2w[answer1]
    #print("The ground truth answer during test was ", groundtruth_answer)
    hk.write("  The ground truth answer during test was " +  groundtruth_answer + '\n')

    if torch.cuda.is_available():
          question, answer = question.cuda(), answer.cuda()

    answer_predicted = model(question.unsqueeze_(0))
    if print_flag:
        print("Shape of predicted answer is ", answer_predicted.shape, " and that of original answer was ", answer.shape)        
    #answer_predicted = answer_predicted.reshape(answer.shape[0], num_answerclasses)
    c = gumbel_argmax(answer_predicted,0)
    c = torch.max(c,-1)[1]
    c = c.cpu().detach().numpy().tolist()
    answer = answer.detach().cpu().numpy()
    predicted_answer = ' '.join(answer_i2w[k] for k in c)
    #print("The predicted answer was ", predicted_answer)
    hk.write("  The predicted during test was " +  predicted_answer + '\n')
    hk.write('\n')
    hk.close()
    return 
    if print_flag:
       print(" Shape of c: ",  c)
    w= []
    for (a,b) in zip(c, answer):
       y_predicted.append(a)
       y_true.append(b)
  print( accuracy_score(y_true, y_predicted))
  return total_loss / ( i + 1 )




def val():
  model.eval()
  total_loss = 0
  y_true = []
  y_predicted = []
  for i, data in enumerate(val_loader):
    question, answer = data[0], data[2]
    question, answer = Variable(question), Variable(answer)
    if print_flag:
        print("Shape of question, caption and answer is ", question.shape,  answer.shape)      
    if torch.cuda.is_available():
          question, answer = question.cuda(), answer.cuda()

    answer_predicted = model(question)
    if print_flag:
        print("Shape of predicted answer is ", answer_predicted.shape, " and that of original answer was ", answer.shape)        
    loss = criterion(answer_predicted.view(-1, num_answerclasses), answer.view(-1))
    total_loss += loss.item()
    answer_predicted = answer_predicted.reshape(answer.shape[0], num_answerclasses)
    c = gumbel_argmax(answer_predicted,0)
    c = torch.max(c,-1)[1]
    c = c.cpu().detach().numpy().tolist()
    answer = answer.detach().cpu().numpy()
    if print_flag:
       print(" Shape of c: ",  c)
    w= []
    for (a,b) in zip(c, answer):
       y_predicted.append(a)
       y_true.append(b)
  #print( accuracy_score(y_true, y_predicted))
  val_acc = str(accuracy_score(y_true, y_predicted))
  hk = open(logfile_name,'a')
  hk.write("Val set accuracy was " +  val_acc + '\n')

  return total_loss / ( i + 1 )


def train():
  model.train()
  curr_step = 0
  total_loss = 0
  for i, data in enumerate(train_loader):
    question, answer = data[0], data[2]
    question, answer = Variable(question), Variable(answer)
    if print_flag:
        print("Shape of question, caption and answer is ", question.shape,  answer.shape)      
    if torch.cuda.is_available():
          question, answer = question.cuda(), answer.cuda()

    answer_predicted = model(question)
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
       test()
       model.train()
      
  return total_loss/(i+1)

for epoch in range(10):
   train_loss = train()
   val_loss = val()
   print("After epoch ", epoch, " train loss: ", train_loss, " val loss: ", val_loss)
