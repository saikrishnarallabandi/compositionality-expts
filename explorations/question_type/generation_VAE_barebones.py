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
parser.add_argument('--generation', action='store_true',
                    help='use generation on whole test')
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
   logits.squeeze_(0)
   logits.squeeze_(0)
   #print(logits.shape)
   return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]



def gen_evaluate(model, data_full, hidden, train_i2w, data_type):
    #print (data_full.size()), "input size for the generation, should be a single sample"
    model.eval()
    kl_loss = 0
    ce_loss = 0
    original_sample = []
    gen_sample = []
    with torch.no_grad():
        data = data_full[:,0:data_full.size(1)-1]
        targets = data_full[:, 1:]
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        data_type =  Variable(data_type).cuda()
        new_input_token = data[:,0].unsqueeze(1) # <SOS> token
        for d in range(data.size(1)):
            original_sample.append(train_i2w[int(data[:,d])])

        while True:
            recon_batch, _, _ = model(new_input_token, None, data_type)
            #output = torch.nn.functional.softmax(recon_batch, dim=2)
            #generated_token = torch.multinomial(output.squeeze(), 1)[0]
            generated_token = gumbel_argmax(recon_batch,0)
            #print "The shpae of generated token is ", generated_token.shape
            #print generated_token.size(), type(generated_token)
            #print (generated_token.size(), generated_token, "is generated token")
            generated_word = train_i2w[int(generated_token.squeeze())]
            gen_sample.append(generated_word)
            new_input_token = Variable(torch.LongTensor(1,1).fill_(int(generated_token.squeeze()))).cuda()
            if generated_word == "<eos>":
                break
            elif len(gen_sample)==25:
                break

        print(' '.join(original_sample)+"\t\t"+' '.join(gen_sample))





if args.generation:

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

    valid_wids = vqa_dataset.get_wids()

    assert (len(valid_wids) == len(train_wids))
    print(len(valid_wids))
    print(valid_wids.get('bot'), valid_wids.get('UNK'), valid_wids.get('?'))

    ntokens = len(train_wids)
    # Load best model
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    # Enumerate through the test data
    for i,a in enumerate(test_loader):
      data_full = a[0]
      data_type = a[1]
      hidden = None
      gen_evaluate(model, data_full, hidden, data_type)
