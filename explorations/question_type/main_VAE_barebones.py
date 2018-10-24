# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
from data_loader_barebones import *
import model_VAE_barebones as model
import generation_VAE_barebones as generation
from logger import *
import logging
import pickle
import json

script_start_time = time.time()

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../../../../data/VQA/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()


log_flag = 1

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

'''
train_file = '/home/ubuntu/projects/multimodal/data/VQA/train2014.questions.txt'
train_set = vqa_dataset(train_file,1,None)
train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn
                         )
train_wids = vqa_dataset.get_wids()

valid_set = vqa_dataset(train_file, 0, train_wids)
valid_loader = DataLoader(valid_set,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn
                         )

test_loader = DataLoader(valid_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate_fn
                         )

# https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
with open('train_loader.pkl', 'wb') as handle:
    pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('valid_loader.pkl', 'wb') as handle:
    pickle.dump(valid_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_loader.pkl', 'wb') as handle:
    pickle.dump(test_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

json.dump(train_wids, open('train_wids.json', 'w')) # https://codereview.stackexchange.com/questions/30741/writing-defaultdict-to-csv-file

'''

with open('train_loader.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader.pkl', 'rb') as handle:
    valid_loader = pickle.load(handle)

with open('test_loader.pkl', 'rb') as handle:
    test_loader = pickle.load(handle)

train_wids = json.load(open('train_wids.json'))

#valid_wids = vqa_dataset.get_wids()
train_i2w =  {i:w for w,i in train_wids.items()}

print("Loaded stuff in ", time.time() - script_start_time)

#assert (len(valid_wids) == len(train_wids))
#print(len(valid_wids))
#print(valid_wids.get('bot'), valid_wids.get('UNK'), valid_wids.get('?'))


ntokens = len(train_wids)
model = model.VAEModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn(recon_x, x, mu, logvar):
    #print("Shapes of recon_x and x are: ", recon_x.shape, x.shape)

    BCE = criterion(recon_x.view(-1,ntokens), x.view(-1))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print("The loss function is returning ", BCE + KLD)
    return KLD, BCE

lr = args.lr


def evaluate():

  model.eval()
  kl_loss = 0
  ce_loss = 0

  with torch.no_grad():

   for i,a in enumerate(valid_loader):

     data_full = a[0]
     data_type =  Variable(a[1]).cuda()
     data = data_full[:,0:data_full.size(1)-1]
     targets = data_full[:, 1:]
     hidden = None
     data = Variable(data).cuda()
     targets = Variable(targets).cuda()

     recon_batch, mu, log_var = model(data, None, data_type)
     kl,ce = loss_fn(recon_batch, targets,mu,log_var)
     loss  = kl + ce

     kl_loss += kl.item()
     ce_loss += ce.item()

  return kl_loss/(i+1) , ce_loss/(i+1)

num_batches = int(len(train_loader.dataset)/args.batch_size)
print("There are ", num_batches, " batches")
#sys.exit()

train_flag = 1

def train():
  global ctr
  global kl_weight_loop
  model.train()
  kl_loss = 0
  ce_loss = 0
  for i,a in enumerate(train_loader):
     ctr += 1

     data_full = a[0]
     data = data_full[:,0:data_full.size(1)-1]
     targets = data_full[:, 1:]
     hidden = None
     #print ("type is", a[1], type(a[1]))
     data_type =  Variable(a[1]).cuda()
     #print (data_type.size(), "is size of the condtion")
     data = Variable(data).cuda()
     targets = Variable(targets).cuda()

     optimizer.zero_grad()
     recon_batch, mu, log_var = model(data, None, data_type)
     kl,ce = loss_fn(recon_batch, targets,mu,log_var)

     #loss  = kl_weight_loop * kl + ce
     loss = kl + ce
     loss.backward()

     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
     optimizer.step()

     kl_loss += kl.item()
     ce_loss += ce.item()


     if i%1000==0:
         print (i,"Batches done, so generating")
         single_train_sample, single_train_sample_type = (torch.LongTensor(train_loader.dataset[0][0]), torch.LongTensor(train_loader.dataset[0][1]))
         single_train_sample = Variable(single_train_sample).cuda()
         single_train_sample_type = Variable(single_train_sample_type).cuda()
         print (single_train_sample.size(), single_train_sample_type.size(), "before generation")
         generation.gen_evaluate(model, single_train_sample, None, train_i2w, single_train_sample_type)
         model.train()

  return kl_loss/(i+1) , ce_loss/(i+1)



logfile_name = 'log_klannealing'
model_name = 'klannealing_thr010.pth'
g = open(logfile_name,'w')
g.close()

best_val_loss = None
ctr = 0
kl_weight = 0.0001
kl_weight = Variable(torch.from_numpy(np.array([kl_weight])), requires_grad=False).cuda().float()
kl_weight_loop = kl_weight

#kl_weight = torch.LongTensor(kl_weight)
#https://math.stackexchange.com/questions/2198864/slow-increasing-function-between-0-and-1

for epoch in range(args.epochs+1):

   #if epoch == 5:
   #   optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
   #   print("Switched to SGD ")
   epoch_start_time = time.time()
   train_klloss, train_celoss = train()
   dev_klloss,dev_celoss = evaluate()
   val_loss = dev_klloss+dev_celoss
   scheduler.step(val_loss)
   #print(time.time() - epoch_start_time)

   # Log stuff
   print("Aftr epoch ", epoch, " Train KL Loss: ", train_klloss, "Train CE Loss: ", train_celoss, "Val KL Loss: ", dev_klloss, " Val CE Loss: ", dev_celoss, "Time: ", time.time() - epoch_start_time)
   g = open(logfile_name,'a')
   g.write("Aftr epoch " + str(epoch) + " Train KL Loss: " + str(train_klloss) + " Train CE Loss: " + str(train_celoss) + " Val KL Loss: " + str(dev_klloss) + " Val CE Loss: " + str(dev_celoss) + " Time: " + str(time.time() - epoch_start_time)  + '\n')
   g.close()

   # Save stuff
   if not best_val_loss or val_loss < best_val_loss:
       with open(model_name, 'wb') as f:
           torch.save(model, f)
       best_val_loss = val_loss
