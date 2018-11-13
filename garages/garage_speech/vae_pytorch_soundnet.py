#from utils import *
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report
from keras.callbacks import *
import pickle, logging
import torch

num_classes = 3
input_dim = 512
hidden = 256
batch_size = 32



# Process labels
labels_file = 'ComParE2018_SelfAssessedAffect.tsv'
labels = {}
ids = ['l','m','h']
f = open(labels_file)
cnt = 0 
for line in f:
  if cnt == 0:
    cnt+= 1
  else:
    line = line.split('\n')[0].split()
    fname = line[0].split('.')[0]
    lbl = ids.index(line[1])
    labels[fname] = lbl
    
#binary2id = {i:w for w,i in labels.iteritems()} python3 stuff
binary2id = {i:w for w,i in labels.items()}


# Process the dev
print("Processing Dev")
f = open('files.devel')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = 'SELFASSESSED_soundnet/' + line + '.npz'
    A = np.load(input_file, encoding = 'latin1')
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    inp = np.float32(inp) 
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])



x_dev = np.array(devel_input_array)
y_dev = to_categorical(devel_output_array,num_classes)
y_dev = np.array(y_dev)


# Process the train
print("Processing Train")
f = open('files.train')
train_input_array = []
train_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = 'SELFASSESSED_soundnet/' + line + '.npz'
    A = np.load(input_file, encoding = 'latin1')
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    inp = np.float32(inp) 
    train_input_array.append(inp)
    train_output_array.append(labels[line])

x_train = np.zeros( (len(train_input_array), 1601, input_dim), 'float32')
y_train = np.zeros( (len(train_input_array), num_classes ), 'float32')

x_train = np.array(train_input_array)
y_train = to_categorical(train_output_array,num_classes)
y_train = np.array(y_train)

# Process the test
print("Processing Test")
f = open('files.devel')
test_input_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = 'SELFASSESSED_soundnet/' + line + '.npz'
    A = np.load(input_file, encoding = 'latin1')
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    inp = np.float32(inp) 
    test_input_array.append(inp)

x_test = np.zeros( (len(test_input_array), 1601, input_dim), 'float32')

for i, x in enumerate(test_input_array):
   x_test[i] = x

train_input = x_train
dev_input = x_dev
train_output = y_train
dev_output = y_dev

def get_uar(epoch):
   y_dev_pred_binary = model.predict(x_dev)
   y_dev_pred = []
   for y in y_dev_pred_binary:
       y_dev_pred.append(np.argmax(y))

   y_dev_ascii = []
   for y in y_dev:
       y_dev_ascii.append(np.argmax(y))

   print ("UAR after epoch ", epoch, " is ")
   print( classification_report(y_dev_ascii, y_dev_pred))
   print(recall_score(y_dev_ascii, y_dev_pred, average='macro'))


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn="print"):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
       pass
       #get_uar(epoch)
       #test(epoch)
       #get_challenge_uar(epoch)

from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import numpy as np
import random
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import sys
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


def loss_fn(recon_x, x, mu, logvar):
    #print("Shapes of recon_x and x are: ", recon_x.shape, x.shape)

    BCE = criterion(recon_x.view(-1,num_classes), x.view(-1))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print("The loss function is returning ", BCE + KLD)
    return KLD, BCE


class vae_model(nn.Module):

   def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
      super(vae_model, self).__init__()

      self.enc_input2hidden = nn.Linear(input_dim,hidden_dim)
      self.enc_hidden2mu =  nn.Linear(hidden_dim, latent_dim)
      self.enc_hidden2sigma =  nn.Linear(hidden_dim, latent_dim)

      self.dec_z2out = nn.Linear(latent_dim, num_classes)

   def reparameterize(self, mu, log_var):
       std = torch.exp(0.5*log_var)
       eps = torch.rand_like(std)
       return eps.mul(std).add_(mu)

   def encoder(self, x):
       x_enc = F.relu(self.enc_input2hidden(x))
       return F.relu(self.enc_hidden2mu(x_enc)), F.relu(self.enc_hidden2sigma(x_enc))

   def forward(self,x):
      mu, log_var = self.encoder(x)
      z = self.reparameterize(mu, log_var)
      return self.dec_z2out(z), mu, log_var

class COMPARE(Dataset):

    def __init__(self, A,B):

        self.input = A
        self.output = B

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        #print (self.input[idx].shape)
        return self.input[idx], self.output[idx]


trainset = COMPARE(train_input_array,train_output_array)
devset = COMPARE(devel_input_array,devel_output_array)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(devset, batch_size=1, shuffle=False)

latent_dim = 3
model = vae_model(input_dim, hidden, latent_dim, num_classes)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def eval():
 kl_loss = 0
 ce_loss = 0
 with torch.no_grad():
   for step, (a,b) in enumerate(dev_loader):
    model.eval()
    #print(a.shape, b.shape)
    a,b = Variable(a), Variable(b)
    a,b = a.cuda(), b.cuda()
    c = model(a)
    #print("Shape of c: ", c.shape)
    recon_batch, mu, log_var = model(a)
    #print("Shape of c: ", c.shape)
    kl,ce = loss_fn(recon_batch, b,mu,log_var)
    kl_loss += kl.item()
    ce_loss += ce.item()
    loss = kl + ce

 return kl_loss/( step + 1), ce_loss/(step + 1)


def test():
 l = 0
 y_true = []
 y_pred = []
 with torch.no_grad():
   for step, (a,b) in enumerate(dev_loader):
    model.eval()
    y_true.append(b.item())
    #print(a.shape, b.shape)
    a,b = Variable(a), Variable(b)
    a,b = a.cuda(), b.cuda()
    c, _,_ = model(a)
    #print("Shape of c: ", c)
    c = c.reshape(c.shape[0],c.shape[1])
    c = torch.max(c,-1)[1]
    c = c.cpu().detach().numpy()
    y_pred.append(c[0])
    #print(" c: ", c)

 print(classification_report(y_true, y_pred))
 print(recall_score(y_true, y_pred, average='macro'))



def train():
  kl_loss = 0
  ce_loss = 0
  for step, (a,b) in enumerate(train_loader):
    model.train()
    #print(a.shape, b.shape)
    a,b = Variable(a), Variable(b)
    a,b = a.cuda(), b.cuda()
    recon_batch, mu, log_var = model(a)
    #print("Shape of c: ", c.shape)
    kl,ce = loss_fn(recon_batch, b,mu,log_var)
    kl_loss += kl.item()
    ce_loss += ce.item()
    loss = kl + ce

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
  return kl_loss/( step + 1), ce_loss/(step + 1)
 

for epoch in range(20):
   training_klloss, training_celoss = train()
   val_klloss, val_celoss = eval()  
   print(epoch, training_klloss, training_celoss, val_klloss, val_celoss)
   test()
