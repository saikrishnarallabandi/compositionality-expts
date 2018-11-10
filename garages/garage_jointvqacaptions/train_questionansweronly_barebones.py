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


vqa_train_gold_file = '/home/ubuntu/projects/multimodal/repos/VQA/PythonHelperTools/gold_annotations_train.csv' # Image_filename, ImageID, QuestionID, Question, Question_Type, Answer_Type, Answer_Majority
vqa_val_gold_file = '/home/ubuntu/projects/multimodal/repos/VQA/PythonHelperTools/gold_annotations_val.csv'
coco_train_captions_file = 'image_caption_train.txt_nospaces'
coco_train_captions_file_nofilenames = 'image_caption_train.txt_nospaces_nofilenames'
image_dir = ''
wids_global = defaultdict(lambda: len(wids_global))


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
   #sys.exit()
   return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]





class jointvqacaptionsdataset(Dataset):

     def __init__(self, vqa_file, captions_file, image_dir, train_flag=1, wids=None):
        self.image_paths, self.image_ids, self.question_ids, self.questions, self.question_types, self.answer_types, self.answers = self.parse_vqafile(vqa_file)

     def parse_vqafile(self, file):
        image_paths = []
        image_ids = []
        question_ids = []
        questions = []
        question_types = []
        answer_types = []
        answers = []
        f = open(file)
        ctr = 0
        for line in f:
           if ctr == 0:
              ctr += 1
              continue
           line = line.split('\n')[0].split('|||')
           image_paths.append(line[0])
           image_ids.append(line[1])
           question_ids.append(line[2])
           questions.append(line[3])
           question_types.append(line[4])
           answer_types.append(line[5])
           answers.append(line[7]) # Taking max instead of union
        return image_paths, image_ids, question_ids, questions, question_types, answer_types, answers



jvcd = jointvqacaptionsdataset(vqa_train_gold_file,coco_train_captions_file,image_dir)
questions_train = jvcd.questions
answers_train = jvcd.answers
#answers_train = jvcd.answer_types
#answertypes_train = jvcd.answer_types

questions_wids = defaultdict(lambda: len(questions_wids))
questions_wids['_PAD'] = 0
questions_wids['UNK'] = 1
trainquestions_ints = []
for q in questions_train: # sorry for varirable names. i am tired. will change later. 
    #print(q)
    l = []
    for k in q.split():
       k_l = questions_wids[k]
       l.append(k_l)
    trainquestions_ints.append(l)
print("Length of question_wids: ", len(questions_wids))

answer_wids = defaultdict(lambda: len(answer_wids))
trainanswer_ints = []
for a in answers_train: # sorry for varirable names. i am tired. will change later. 
    l = []
    #for k in a.split():
    #   k_l = answer_wids[k]
    #   l.append(k_l)
    k_l = answer_wids[a.split()[0]]      
    trainanswer_ints.append(k_l)
print("Length of answer_wids: ", len(answer_wids))


assert len(trainquestions_ints) == len(trainanswer_ints) 


jvcd = jointvqacaptionsdataset(vqa_val_gold_file,coco_train_captions_file,image_dir)
questions_val = jvcd.questions
answers_val = jvcd.answers
answers_val = jvcd.answer_types

valquestions_ints = []
for q in questions_val: # sorry for varirable names. i am tired. will change later.
    l = []
    if a.split()[0] in answer_wids:
       k_l = answer_wids[a.split()[0]]      
    else:
       k_l = 1
    valquestions_ints.append(l)
print("Length of question_wids: ", len(questions_wids))

valanswer_ints = []
for a in answers_val: # sorry for varirable names. i am tired. will change later. 
    l = []
    #for k in a.split():
    #   k_l = answer_wids[k]
    #   l.append(k_l)
    if a.split()[0] in answer_wids:
       k_l = answer_wids[a.split()[0]]      
    else:
       k_l = 1
    valanswer_ints.append(k_l)

assert len(valquestions_ints) == len(valanswer_ints) 
print("Length of answer_wids: ", len(answer_wids))
print("Length of question_wids: ", len(questions_wids))
#sys.exit()

'''
np.save('train_questions.npy', trainquestions_ints)
np.save('train_answers.npy', trainanswer_ints)
np.save('val_questions.npy', valquestions_ints)
np.save('val_answers.npy', valanswer_ints)

with open('question_wids.json', 'w') as outfile:
            json.dump(questions_wids, outfile)
with open('answers_wids.json', 'w') as outfile:
            json.dump(answer_wids, outfile)
'''

question_i2w =  {i:w for w,i in questions_wids.items()}
answer_i2w = {i:w for w,i in answer_wids.items()}

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


jvcdset = jointvqacaptions_dataset(trainquestions_ints, trainquestions_ints, trainanswer_ints)
train_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )
num_questionclasses = len(questions_wids)
num_answerclasses = len(answer_wids)

jvcdset = jointvqacaptions_dataset(valquestions_ints, valquestions_ints, valanswer_ints)
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
      self.lstm = nn.LSTM(question_embed_dim, hidden_dim, batch_first = True, bidirectional=True)
      self.hidden2out =  nn.Linear(hidden_dim*2, num_answerclasses)

   def forward(self,question):
      question_embedding = self.question_embedding(question)

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
  print( accuracy_score(y_true, y_predicted))
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
       #model.train()
       print(" Train Loss after ", i , " updates: ", total_loss/(i+1))
  return total_loss/(i+1)

for epoch in range(30):
   train_loss = train()
   val_loss = val()
   print("After epoch ", epoch, " train loss: ", train_loss, " val loss: ", val_loss)
