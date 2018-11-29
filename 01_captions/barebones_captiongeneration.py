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
import time

print_flag = 0
logfile_name = 'log_modelpredictions_image2captions'
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
#imageid2features.pkl
with open('imageid2captions.pkl', 'rb') as f:
   imageid2captions = pickle.load(f)

trainquestion_ints = []
trainanswer_ints = []
trainimage_ids = []
trainfeatures = []
traincaption_ints = []
h = open('answers_vqa2014_jacobandreassplit_image_caption.txt','w')
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
for v in val_dict:
 try:
   if len(v.keys()) > 0:
    question = v['question_str']
    answer = v['valid_answers'][0]
    image_id = v['image_name'] + '.jpg'
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

      def __init__(self, questions, features, answers, captions):
         self.questions = questions
         self.features = features
         self.answers = answers
         self.captions = captions

      def __len__(self):
        return len(self.features)

      def __getitem__(self, item):
        return self.features[item],  self.captions[item]

      def collate_fn(self, batch):

       caption_lengths = [len(x[1]) for x in batch]
       max_len = np.max(caption_lengths) + 1

       captions =  np.array([ self._pad(x[1], max_len)  for x in batch ], dtype=np.int)
       features = [x[0] for x in batch]

       captions = torch.LongTensor(captions)
       features = torch.FloatTensor(features)

       return features, captions

      def _pad(self, seq, max_len):
        return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


jvcdset = jointvqacaptions_dataset(trainquestion_ints, trainfeatures, trainanswer_ints, traincaption_ints)
train_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )
num_questionclasses = len(questions_wids)
num_answerclasses = len(answer_wids)


jvcdset = jointvqacaptions_dataset(valquestion_ints, valfeatures, valanswer_ints, valcaption_ints)
val_loader = DataLoader(jvcdset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )



class baseline_model(nn.Module):

   def __init__(self, question_embed_dim, caption_embed_dim, answer_embed_dim, num_classes, num_answerclasses):
      super(baseline_model, self).__init__()
      hidden_dim = 128
      resnetfeats_dim = 2048
      self.question_embedding = nn.Embedding(num_classes, question_embed_dim)
      self.caption_embedding = nn.Embedding(num_classes, caption_embed_dim)
      self.embed2lstm_question = nn.Linear(question_embed_dim,question_embed_dim//2) # 128 -> 64
      self.embed2lstm_caption = nn.Linear(caption_embed_dim,caption_embed_dim//2) # 128 -> 64
      self.lstm = nn.LSTM(caption_embed_dim//2 +1024, hidden_dim, 2, batch_first = True, bidirectional=True)
      self.hidden2out =  nn.Linear(hidden_dim*2, num_questionclasses)
      self.dropout = nn.Dropout(0.3)
      self.image_embedding = nn.Linear(resnetfeats_dim, 1024) # 2048 -> 1024

   def forward(self, features, captions):
      captions = captions[:,:-1]
      if print_flag:
          print("Captions size: ", captions.size()) 

      caption_embedding = self.caption_embedding(captions.long())
      caption_embedding = F.relu(self.embed2lstm_caption(caption_embedding))
      caption_embedding = self.dropout(caption_embedding)

      if print_flag:
         print("Shape of features: ", features.shape)

      # Repeat image feature embedding and concatenate
      features = F.relu(self.image_embedding(features.float()))
      features.unsqueeze_(1)
      length_desired = captions.shape[1]
      features = features.repeat(1, length_desired, 1) 

      if print_flag:
         print("Shape of caption embedding: ", caption_embedding.shape)
         print("Shape of features: ", features.shape)

      total_embedding = torch.cat([features, caption_embedding], dim=-1)     
      if print_flag:
         print("Shape of total embedding: ", total_embedding.shape)

      x, hidden = self.lstm(total_embedding)

      if print_flag:
         print("Shape of lstm output: ", x.shape)
      return self.hidden2out(x)


   def sample(self, features, states=None, max_len=20):
        """Accept a pre-processed image tensor (inputs) and return predicted 
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        """
 
        if print_flag:
           print("In sample")
           print("Shape of my input: ", features.shape)
        features = features.unsqueeze(0)
        features = F.relu(self.image_embedding(features.float()))
        features.unsqueeze_(1)
        length_desired = max_len
        features = features.repeat(1, length_desired, 1) 
        if print_flag:
           print("Shape of features is ", features.shape)

        inputs = features.new(features.shape[0], 1, 1)
        inputs.zero_()

        sampled_ids = []
        for i in range(max_len):

            caption_embedding = self.caption_embedding(inputs.long())
            caption_embedding = F.relu(self.embed2lstm_caption(caption_embedding))
            caption_embedding = caption_embedding.squeeze(0)

            if print_flag:
               print("Shape of features before unsqueezing and caption embedding after squeezing: ", features[:,i,:].shape, caption_embedding.shape)
            total_embedding = torch.cat([features[:,i,:].unsqueeze_(1), caption_embedding], dim=-1)

            hiddens, states = self.lstm(total_embedding, states)
            outputs = self.hidden2out(hiddens.squeeze(1))

            predicted = gumbel_argmax(outputs,0)
            predicted = predicted.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = predicted.unsqueeze(0)
            inputs = inputs.unsqueeze(0)

        return sampled_ids


question_embed_dim = 128
caption_embed_dim = 128
answer_embed_dim = 128

model = baseline_model(question_embed_dim, caption_embed_dim, answer_embed_dim, num_questionclasses, num_answerclasses)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def gen():
  model.eval()
  for i, data in enumerate(val_loader):
    features, captions = data[0], data[1]
    features, captions = Variable(features), Variable(captions)
    if print_flag:
       print("Shape of features during gen is ", features.shape)
       print("Shape of captions during gen is ", captions.shape)
   
    if torch.cuda.is_available():
          features, caption = features.cuda(), captions.cuda()
    features = features[0]
    captions = captions[0].detach().cpu().numpy()
    captions_predicted = model.sample(features)
    #print("I predicted: ", ' '.join(question_i2w[k] for k in captions_predicted) + '\n')
    #print("I predicted: ", ' '.join(question_i2w[k] for k in captions_predicted) + ' while the original caption was ' + ' '.join(question_i2w[k] for k in captions) + '\n')     
    hk = open(logfile_name,'a')
    hk.write("Original Caption: " + ' '.join(question_i2w[k] for k in captions) + '\n')
    hk.write("I predicted: " + ' '.join(question_i2w[k] for k in captions_predicted) + '\n')
    hk.write('\n')
    hk.close()
    return 

def train():
  model.train()
  curr_step = 0
  total_loss = 0
  for i, data in enumerate(train_loader):
      #print(i, data[0].shape, data[1].shape)
      features, captions = data[0], data[1]
      features, captions = Variable(features), Variable(captions)
      if torch.cuda.is_available():
          features, captions = features.cuda(), captions.cuda()
      captions_predicted = model(features, captions)
      captions = captions[:,1:]
      #print("Shape of predicted and original captions: ", captions.shape, captions_predicted.shape)
      
      loss = criterion(captions_predicted.contiguous().view(-1, num_questionclasses), captions.contiguous().view(-1))
      total_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
      optimizer.step()

      if i % 1000 == 1:
         print(" Train Loss after ", i , " updates: ", total_loss/(i+1))
         hk = open(logfile_name,'a')
         hk.write(" Train Loss after " + str(i) + " batches: " + str(total_loss/(i+1)) + '\n')
         hk.close()
         gen()
         model.train()

for epoch in range(10):
   epoch_start_time = time.time()
   train_loss = train()
   hk = open(logfile_name,'a')
   hk.write("Train Loss after epoch " + str(epoch)  + " " + str(train_loss) + '\n')
   hk.close()
