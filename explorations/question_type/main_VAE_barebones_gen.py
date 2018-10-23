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
from logger import *
import logging
import gen_VAE_barebones as gen_evaluate


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
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
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

valid_wids = vqa_dataset.get_wids()

assert (len(valid_wids) == len(train_wids))
print(len(valid_wids))
print(valid_wids.get('bot'), valid_wids.get('UNK'), valid_wids.get('?'))


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
     data = data_full[:,0:data_full.size(1)-1]
     targets = data_full[:, 1:] 
     hidden = None
     data = Variable(data).cuda()
     targets = Variable(targets).cuda()

     recon_batch, mu, log_var = model(data, None)
     kl,ce = loss_fn(recon_batch, targets,mu,log_var)
     loss  = kl + ce

     kl_loss += kl.item()
     ce_loss += ce.item()

  return kl_loss/i , ce_loss/i 

num_batches = int(len(train_loader.dataset)/args.batch_size)
print("There are ", num_batches, " batches")
#sys.exit()

def train():
  global ctr
  global kl_weight
  model.train()
  kl_loss = 0
  ce_loss = 0
  for i,a in enumerate(train_loader):
     ctr += 1
     if i < 1:
       print(a[0].shape,a[1].shape, i)
     data_full = a[0]
     data = data_full[:,0:data_full.size(1)-1]
     targets = data_full[:, 1:]
     hidden = None
     data = Variable(data).cuda()
     targets = Variable(targets).cuda()
     #hidden = repackage_hidden(hidden)
     model.zero_grad()
     recon_batch, mu, log_var = model(data, None)
     kl,ce = loss_fn(recon_batch, targets,mu,log_var)
     #loss  = kl_weight * kl + ce
     loss = kl_weight + ce

     optimizer.zero_grad()
     loss.backward()
     #optimizer.step()

     #if ctr % 1000 == 1:
     #   kl_weight += 0.1 
     #   print("KL Weight now is ", kl_weight)

     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
     optimizer.step()
     #for p in model.parameters():
     #    p.data.add_(-lr, p.grad.data)

     kl_loss += kl.item()
     ce_loss += ce.item()
     # generate at every 1000 batches:
     if i%1000==0:
         print (i,"Batches done, so generating")
         single_train_sample, single_train_sample_type = (train_loader.dataset[0][0], train_loader.dataset[0][1])
         gen_evaluate(single_train_sample, single_train_sample_type, train_test=True)   
         #single_train_sample = [train_loader[330][0][0].unsqueeze(0)]
         #single_train_sample_type = [train_loader[330][0][1].unsqueeze(0)]
         #print( single_train_sample.size(), single_train_sample_type.size())
         #single_train_sample_type = [corpus.train_type[330][0].unsqueeze(0)]
         #gen_evaluate(single_train_sample, single_train_sample_type, train_test=True)         
     # Check for Nan
     if torch.isnan(loss):
        print("Encountered Nan value. exiting")
        sys.exit()

  return kl_loss/i , ce_loss/i 


logfile_name = 'log_barebones_thr010_klannealing'
model_name = 'barebones_thr010_klannealing.pth'
g = open(logfile_name,'w')
g.close()

best_val_loss = None
ctr = 0
kl_weight = 0

for epoch in range(args.epochs+1):

   #if epoch == 5:
   #   optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
   #   print("Switched to SGD ") 
   epoch_start_time = time.time()
   train_klloss, train_celoss = train()
   dev_klloss,dev_celoss = evaluate()
   val_loss = dev_klloss+dev_celoss
   scheduler.step(val_loss)

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

