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
ntype_emb = 30

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
model = model.RNNModel(args.model, ntype_emb,  ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=corpus.PAD_IDX)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data).cuda()
    else:
        return tuple(repackage_hidden(v) for v in h)



def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)

    #with torch.no_grad():
    for i in range(0, len(data_source), args.bptt):
        data_full = data_source[i]
        data_type =
        data = data_full[:,0:data_full.size(1)-1]
        targets = data_full[:, 1:]
        #hidden = model.init_hidden(data.size(0))
        hidden = None
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets.contiguous().view(-1)).data
        #hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for i in range(0, len(corpus.train), args.bptt):
        data_full = corpus.train[i]
        data = data_full[:,0:data_full.size(1)-1]
        targets = data_full[:, 1:]
        data_type = corpus.train_type[i]
        print data_type.size(), data_full.size()

        #print data.size(), targets.size()
        hidden = None
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        #hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, None, data_type)
        #print output.size(), targets.size()
        loss = criterion(output.view(-1, ntokens), targets.contiguous().view(-1))
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        #print loss
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            #print(epoch, type(epoch))
            print("epoch, loss, ppl", epoch, cur_loss[0], math.exp(cur_loss[0]))
            #print('| epoch {:3d} | {:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, lr,0, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        #print("in train and val")
        val_loss = evaluate(corpus.valid)
        print('-' * 89)
        #print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                   val_loss, math.exp(val_loss)))
        print("end epoch, loss, ppl", epoch, val_loss[0], math.exp(val_loss[0]))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss[0] < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss[0]
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
