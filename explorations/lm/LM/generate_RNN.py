


# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
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

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, 32)
print len(corpus.train), "Batches for Train ||| ", len(corpus.train)*32, "Samples of Train"
print len(corpus.valid), "Batches for Valid ||| ", len(corpus.valid)*32, "Samples of Valid"
print len(corpus.test), "Samples for Test"

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=corpus.PAD_IDX)

###############################################################################
# Generation code
###############################################################################



def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    #with torch.no_grad():
    for i in range(0, len(data_source), args.bptt):
        data_full = data_source[i]
        data = data_full[:,0:data_full.size(1)-1]
        targets = data_full[:, 1:]
        hidden = None
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        #input_token  = corpus.dictionary.word2idx["<sos>"]
        original_questions = []
        gen_questions = []
        new_input_token = data[:,0].unsqueeze(1) # SOS
        for d in range(data.size(1)):
            original_questions.append(corpus.dictionary.idx2word[int(data[:,d])])
        while True:
            #input_token = data[:,d].unsqueeze(1)
            output, hidden = model(new_input_token, hidden)
            ## SAMPLE
            output = torch.nn.functional.softmax(output, dim=2)
            generated_token = torch.multinomial(output.squeeze(), 1)[0]
            #print generated_token.size(), type(generated_token)
            generated_word = corpus.dictionary.idx2word[int(generated_token.squeeze())]
            gen_questions.append(generated_word)
            new_input_token = Variable(torch.LongTensor(1,1).fill_(int(generated_token.squeeze()))).cuda()

            ###ARGMAX
            """
            _, generated_token = torch.max(output.detach(), 2)
            generated_token = int(generated_token.data[0])
            generated_word = corpus.dictionary.idx2word[generated_token]
            new_input_token = Variable(torch.LongTensor(1,1).fill_(generated_token)).cuda()
            input_token = corpus.dictionary.idx2word[generated_token]
            gen_questions.append(generated_word)"""
            if generated_word == "<eos>":
                break
            elif len(gen_questions)==25:
                break

        print ' '.join(original_questions)+"\t\t"+' '.join(gen_questions)

        #hidden = model.init_hidden(data.size(0))
    return None


# Loop over epochs.
lr = args.lr
best_val_loss = None

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.

test_loss = evaluate(corpus.test)
