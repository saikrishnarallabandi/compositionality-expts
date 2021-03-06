# coding: utf-8
import argparse
import time
import math
import os, sys
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
import data
import model
from logger import *

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../../data',
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
parser.add_argument('--epochs', type=int, default=50,
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
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--data_percent', type=float,default=1.0,
                    help='amount of data used to train LM')
parser.add_argument('--results_file', type=str,default='results.txt')
args = parser.parse_args()

logger = Logger(args.logdir)

if not os.path.exists(os.path.dirname(args.results_file)):
     os.makedirs(os.path.dirname(args.results_file))

g = open(args.results_file,'w')
g.close()

#sys.exit()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.cuda()

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    if isinstance(h, torch.Tensor):
        #print "here"
        return h.detach()
    else:
        #print "here"
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source,epoch):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    #with torch.no_grad():
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i)
        data = Variable(data)
        #hidden = Variable(hidden)
        targets = Variable(targets)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    logger.scalar_summary('Val Perplexity per epoch', math.exp(total_loss.item()/len(data_source)) , epoch)
    return total_loss / len(data_source)


def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    #print("in train")
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
      if batch*args.batch_size/train_data.size(0) < args.data_percent:
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        #print("got batch")
        hidden = repackage_hidden(hidden)
        #print "hidden"
        model.zero_grad()
        data = Variable(data)
        #hidden = Variable(hidden)
        targets = Variable(targets)
        output, hidden = model(data, None)
        #print("model formward")
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        #print loss
        #if batch % args.log_interval == 0 and batch > 0:
        #    cur_loss = total_loss / args.log_interval
        #    elapsed = time.time() - start_time
            #print(epoch, type(epoch))
        #    print("epoch, loss, ppl", epoch, cur_loss.item(), math.exp(cur_loss.item()))
        #    #print('| epoch {:3d} | {:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, lr,0, cur_loss, math.exp(cur_loss)))
        #    total_loss = 0
        #    start_time = time.time()

      else:
         return total_loss.item()/(batch+1)

    #logger.scalar_summary('Train Perplexity per epoch', math.exp(total_loss.item()/(batch+1)) , epoch) 

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).cuda()
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(epoch)
        #print("in train and val")
        #print("In epoch ", epoch, " train loss: ", train_loss, "train ppl: ", math.exp(train_loss))
        val_loss = evaluate(val_data,epoch)
        #print('-' * 89)
        print("Epoch: ", epoch, " Train Loss: ", train_loss, " Train ppl: ", math.exp(train_loss), " Val loss: ", val_loss.item(), " Val ppl: ", math.exp(val_loss.item()))
        g = open(args.results_file,'a')
        g.write("Epoch: " +  str(epoch) + " Train Loss: " +  str(train_loss) + " Train ppl: " +  str(math.exp(train_loss)) + " Val loss: " + str(val_loss.item()) +  " Val ppl: " + str(math.exp(val_loss.item())) + '\n')
        g.close()

        #print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss.item() < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss.item()
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
'''
test_loss = evaluate(test_data,10)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
'''
