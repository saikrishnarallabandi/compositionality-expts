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

from pytorch_compact_bilinear_pooling.compact_bilinear_pooling import CountSketch
from pytorch_compact_bilinear_pooling.compact_bilinear_pooling import CompactBilinearPooling

print_flag = 0
logfile_name = "mcb_testing.txt"
hk = open(logfile_name,'w')
hk.close()
remove_cap = True
remove_img = False

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

#with open('qid2qtype.pkl', 'rb') as f:
with open('qid2atpye.pkl', 'rb') as f:
   qid2qtype = pickle.load(f)

# sanity check
print('248349 ', qid2qtype['248349'])

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


jvcdset = jointvqacaptions_dataset(valquestion_ints, valfeatures, valanswer_ints, valcaption_ints, valquestion_ids)
val_loader = DataLoader(jvcdset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=jvcdset.collate_fn
                         )



class baseline_model(nn.Module):

   def __init__(self, question_embed_dim, caption_embed_dim, answer_embed_dim, num_classes, num_answerclasses):
      super(baseline_model, self).__init__()
      hidden_dim = 256
      resnetfeats_dim = 2048
      self.num_latentdim = 64
      self.question_embedding = nn.Embedding(num_classes, question_embed_dim)
      self.caption_embedding = nn.Embedding(num_classes, caption_embed_dim)
      self.embed2lstm_question = nn.Linear(question_embed_dim,question_embed_dim*2)
      self.embed2lstm_caption = nn.Linear(caption_embed_dim,caption_embed_dim*2)
      self.lstm = nn.LSTM(question_embed_dim*2, hidden_dim, 2, batch_first = True, bidirectional=True)
      self.dropout = nn.Dropout(0.3)
      self.image_embedding = nn.Linear(resnetfeats_dim,question_embed_dim*2)
 
      self.latent2hidden = nn.Linear(self.num_latentdim, hidden_dim)
      self.hidden2out = nn.Linear(hidden_dim, num_answerclasses)
     
      self.input_size1 = 256
      self.input_size2 = 2048
      self.output_size = 16000

      self.mcb = CompactBilinearPooling(self.input_size1, self.input_size2, self.output_size).cuda()
      self.fc1 = nn.Linear(self.output_size, 8000)
      self.dropout2 = nn.Dropout(0.5)

      self.hidden2mu =  nn.Linear(8000, self.num_latentdim)
      self.hidden2log_var = nn.Linear(8000, self.num_latentdim)


   def forward(self,question, features, captions):
      question_embedding = self.question_embedding(question)
      question_embedding = F.relu(self.embed2lstm_question(question_embedding))
      question_embedding = self.dropout(question_embedding)

      x, hidden = self.lstm(question_embedding)       
      question_ctx = hidden[0][-1]
      
      mcb_embedding = self.mcb(question_ctx, features)

      # Get mu and sigma
      x = self.dropout2(self.fc1(mcb_embedding))

      mu = self.hidden2mu(x)
      sigma = self.hidden2log_var(x)

      # Reparameterize
      std = torch.exp(0.5*sigma)
      eps = torch.rand_like(std)
      z = eps.mul(std).add_(mu)

      # Decoder
      x = F.relu(self.latent2hidden(z))
      x = self.hidden2out(x)

      return x, mu, sigma

def loss_fn(recon_x, x, mu, logvar):

    BCE = criterion(recon_x.view(-1,num_classes), x.view(-1))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD, BCE

def kl_anneal_function(step, k=0.0025, x0=2500, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)


      
updates = 0
question_embed_dim = 128
caption_embed_dim = 128
answer_embed_dim = 128
num_classes = num_answerclasses

model = baseline_model(question_embed_dim, caption_embed_dim, answer_embed_dim, num_questionclasses, num_answerclasses)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def gen():
  model.eval()
  total_loss = 0

  for i, data in enumerate(val_loader):

    question, features, answer, caption = data[0], data[1], data[2], data[3]
    question, features, answer, caption = Variable(question[0,:]), Variable(features[0]), Variable(answer[0]), Variable(caption[0,:])
    if remove_cap:
        caption = caption.data.fill_(1)
        #print(caption)
    if remove_img:
        features = features.data.fill_(1)

    question_id = data[4]

    
    question1 = question
    answer1 = answer.item()
    caption1 = caption

    groundtruth_question = ' '.join(question_i2w[k] for k in question1.detach().numpy())
    groundtruth_caption = ' '.join(question_i2w[k] for k in caption1.detach().numpy())

    hk = open(logfile_name,'a')
    #hk.write("  The ground truth question during test was " +  groundtruth_question + '\n')
    #hk.write("  The ground truth caption during test was " +  groundtruth_caption + '\n')
    groundtruth_answer = answer_i2w[answer1]
    print(question_id.numpy())
    groundtruth_qtype = qid2qtype[str(question_id.numpy()[0])]
    #print(aid2atype[str(answer_id.numpy()[0])])
    #hk.write(groundtruth_answer+" | "+predicted_answer + " | "+ groundtruth_qtype+"\n")
    #hk.write("  The ground truth answer during test was " +  groundtruth_answer + '\n')
    #hk.write("  The ground truth q type during test was " +  groundtruth_qtype + '\n')
    if print_flag:
        print("Shape of question, features, caption and answer is ", question.shape, features.shape, caption.shape, answer.shape)

    if torch.cuda.is_available():
          question, features, answer, caption = question.cuda(), features.cuda(), answer.cuda(), caption.cuda()

    answer_predicted,_,_ = model(question.unsqueeze_(0), features.unsqueeze_(0), caption.unsqueeze_(0))

    c = gumbel_argmax(answer_predicted,0)
    c = torch.max(c,-1)[1]
    c = c.cpu().detach().numpy().tolist()
    answer = answer.detach().cpu().numpy()
    predicted_answer = ' '.join(answer_i2w[k] for k in c)
    #print("The predicted answer was ", predicted_answer)
    #hk.write("  The predicted answer during test was " +  predicted_answer  + '\n')
    #hk.write("  The ground truth q type was " + str(groundtruth_qtype)   + '\n')
    #hk.write('\n')
    hk.write(groundtruth_answer+" | "+predicted_answer + " | "+ groundtruth_qtype+"\n")
    hk.close()
    # Check with Nidhi for the required output format
    #hk.write(" #Nidhi: " + groundtruth_question + ' ' + groundtruth_answer + ' ' +  str(predicted_answer) + ' ' + str(groundtruth_qtype) + '\n') 
   
    #print("I am here")
    #return 


def val():
  model.eval()
  total_loss = 0
  y_true = []
  y_predicted = []
  kl_loss = 0
  ce_loss = 0
  for i, data in enumerate(val_loader):
    question, features, answer, caption = data[0], data[1], data[2], data[3]
    question, features, answer, caption = Variable(question), Variable(features), Variable(answer), Variable(caption)
    if remove_cap:
        caption = caption.data.fill_(1)
        #print(caption)
    if remove_img:
        features = features.data.fill_(1)

    if print_flag:
        print("Shape of question, caption and answer is ", question.shape,  answer.shape, caption.shape)
    if torch.cuda.is_available():
          question, features, answer, caption = question.cuda(), features.cuda(), answer.cuda(), caption.cuda()

    answer_predicted, mu, sigma = model(question, features, caption)
    if print_flag:
        print("Shape of predicted answer is ", answer_predicted.shape, " and that of original answer was ", answer.shape)        
    kl, ce = loss_fn(answer_predicted, answer, mu, sigma)
    loss = kl + ce
    kl_loss += kl.item()
    ce_loss += ce.item()
    total_loss += loss.item()
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
  val_acc = str(accuracy_score(y_true, y_predicted))
  hk = open(logfile_name,'a')
  hk.write("Val set accuracy was " +  val_acc + '\n')   
 
    #if i % 1000 == 1:
       #val_loss = val()
    #   print(" Val Loss after ", i , " updates: ", total_loss/(i+1))

  return total_loss/(i+1)


def train():
  model.train()
  curr_step = 0
  total_loss = 0
  kl_loss = 0
  ce_loss = 0
  global updates

  for i, data in enumerate(train_loader):
    question, features, answer, caption = data[0], data[1], data[2], data[3]
    #print (data[1].size(), data[3].size())
    #exit()
    question, features, answer, caption = Variable(question), Variable(features), Variable(answer), Variable(caption)
    if remove_cap:
        caption = caption.data.fill_(1)
        #print(caption)
    if remove_img:
        features = features.data.fill_(1)
    if print_flag:
        print("Shape of question, caption and answer is ", question.shape,  answer.shape, caption.shape)
    if torch.cuda.is_available():
          question, features, answer, caption = question.cuda(), features.cuda(), answer.cuda(), caption.cuda()

    answer_predicted, mu, log_var = model(question, features, caption)
    if print_flag:
        print("Shape of predicted answer is ", answer_predicted.shape, " and that of original answer was ", answer.shape)        
    kl, ce = loss_fn(answer_predicted, answer, mu, log_var)
    weight = kl_anneal_function(updates)
    kl = kl * weight
    loss = kl + ce
    kl_loss += kl.item()
    ce_loss += ce.item()

    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

    if i % 1000 == 1:
       #gen()
       #val_loss = val()
       model.train()
       print(" Train Loss, KL, CE  after ", i , " updates: ", total_loss/(i+1), kl_loss/(i+1), ce_loss/(i+1))
       #sys.exit()   
  return total_loss/(i+1)

for epoch in range(30):

   train_loss = train()
   val_loss = val()
   print("After epoch ", epoch, " train loss: ", train_loss, " and val loss is  ", val_loss)

   if (epoch+1) % 5 == 0:
     torch.save({"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch
                }, "nocap_VED_model_questionimage_caption.pkl")





#checkpoint = torch.load("model_questionimage_caption.pkl")
#model.load_state_dict(checkpoint['model'])
#model.load_state_dict(torch.load('nocap_model_questionimage_caption.pkl'))
gen()



