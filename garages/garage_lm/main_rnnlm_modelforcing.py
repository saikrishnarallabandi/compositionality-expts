# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
import model_rnnlm_modelforcing as model
from data_loader_barebones import *
#from logger import *
import logging
import pickle
import json
import numpy as np
import generation_rnn as generation
import gc
import random

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
lr = args.lr

log_flag = 1
generation_flag = 1

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################


train_file = '/home/ubuntu/projects/multimodal/data/VQA/train2014.questions.txt'
valid_file = '/home/ubuntu/projects/multimodal/data/VQA/val2014.questions.txt'
train_set = vqa_dataset(train_file,1,None)
train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn
                         )
train_wids = vqa_dataset.get_wids()

valid_set = vqa_dataset(valid_file, 0, train_wids)
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

#sys.exit()


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
print("Number of tokens is ", ntokens)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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

     output, hidden = model(data, None)
     loss = criterion(output.view(-1, ntokens), targets.view(-1))

     ce_loss += ce.item()

  return  ce_loss/(i+1)

num_batches = int(len(train_loader.dataset)/args.batch_size)
print("There are ", num_batches, " batches")
#sys.exit()

train_flag = 1

max_length = 32

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

pid = os.getpid()
teacher_forcing_ratio = 0.98

def train():
  global ctr
  global teacher_forcing_ratio
  model.train()
  ce_loss = 0
  hidden = model.init_hidden(args.batch_size)
  prev_mem = 0
  for i,a in enumerate(train_loader):
     ctr += 1
     hidden = repackage_hidden(hidden)
     #ctr += 1
     #print(i)  
     use_teacher_forcing = random.random() < teacher_forcing_ratio
     data_full = a[0]
     del a 
     len_data = data_full.size(1)
     length_desired = max_length if len_data > max_length else len_data
     data = data_full[:,0:length_desired-1]
     targets = data_full[:, 1:length_desired+1]
     data = Variable(data,requires_grad=False).cuda()
     targets = Variable(targets,requires_grad=False).cuda()
     #optimizer.zero_grad()
     model.zero_grad()

     if use_teacher_forcing:
       #c = net(a.double())
       output, hidden = model(data, None)
     else:
       output, hidden = model.forward_sample(data)
       #c = net.model_sample(a).cuda()

     output, hidden  = model(data, None)
     #print("Shape of output, data and target: ", output.shape, data.shape, targets.shape, length_desired)
     loss = criterion(output.view(-1, ntokens), targets.view(-1))
     loss.backward()
     ce_loss += loss.item()

     del output, targets, data, loss, len_data, length_desired 

     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
     #optimizer.step()
     #optimizer.zero_grad() 
     for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

     #try:
     #  for obj in gc.get_objects():
     #    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
     #      print(type(obj), obj.size())
     #except OSError:
     #  pass
   
     #for name, param in model.named_parameters():
     #   if param.requires_grad:
     #      print ( name, param.data.shape)

     if i% 100 == 1:
       # Log stuff
       g = open(logfile_name,'a')
       g.write("   Aftr step " + str(i) + " Train CE Loss: " + str(ce_loss/(i+1)) + " Time: " + str(time.time() - epoch_start_time)  + '\n')
       g.close()

     if ctr % 2000 == 1 and teacher_forcing_ratio > 0.1:
          teacher_forcing_ratio -= 0.05

       # https://github.com/pytorch/pytorch/issues/3665 
       #cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
       #add_mem = cur_mem - prev_mem
       #prev_mem = cur_mem
       #print("Added memory: ", add_mem)


     if i%100==0 and generation_flag:
         print (i,"Batches done, so generating")
         single_train_sample, single_train_sample_type = (torch.LongTensor(train_loader.dataset[0][0]).unsqueeze(0), torch.LongTensor([train_loader.dataset[0][1]]).unsqueeze(0))
         single_train_sample = Variable(single_train_sample).cuda()
         single_train_sample_type = Variable(single_train_sample_type).cuda()
         print (single_train_sample.size(), single_train_sample_type.size(), "before generation")
         generation.gen_evaluate(model, single_train_sample, None, train_i2w)
         model.train()

  return ce_loss/(i+1)



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
   train_celoss = train()
   dev_celoss = evaluate()
   val_loss = dev_celoss
   #scheduler.step(val_loss)
   #print(time.time() - epoch_start_time)

   # Log stuff
   g = open(logfile_name,'a')
   g.write("Aftr epoch " + str(epoch) + " Train CE Loss: " + str(train_celoss) + " Val CE Loss: " + str(dev_celoss) + " Time: " + str(time.time() - epoch_start_time)  + '\n')
   g.close()

   # Save stuff
   if not best_val_loss or val_loss < best_val_loss:
       with open(model_name, 'wb') as f:
           torch.save(model, f)
       best_val_loss = val_loss

