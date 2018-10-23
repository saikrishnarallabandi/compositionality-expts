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
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
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

print(valid_wids.get('bot'), valid_wids.get('UNK'), valid_wids.get('?'))


ntokens = len(train_wids)
model = model.VAEModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)

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

 
def train():

  model.train()
  kl_loss = 0
  ce_loss = 0
  for i,a in enumerate(train_loader):
   if i < 1:
     #print(a[0].shape,a[1].shape, i)     
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
     loss  = kl + ce
     loss.backward()

     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
     torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
     for p in model.parameters():
         p.data.add_(-lr, p.grad.data)

     kl_loss += kl.item()
     ce_loss += ce.item()

  return a, kl_loss/i , ce_loss/i 



for epoch in range(args.epochs+1):
   epoch_start_time = time.time()
   a, train_klloss, train_celoss = train()
   print(a)
   dev_klloss,dev_celoss = evaluate()
   val_loss = dev_klloss+dev_celoss
   print(val_loss, epoch, time.time() - epoch_start_time)
