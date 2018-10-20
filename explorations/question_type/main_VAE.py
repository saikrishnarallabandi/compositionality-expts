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
print (len(corpus.train), "Batches for Train ||| ", len(corpus.train)*32, "Samples of Train")
print (len(corpus.valid), "Batches for Valid ||| ", len(corpus.valid)*32, "Samples of Valid")
print (len(corpus.test), "Samples for Test")

ntokens = len(corpus.dictionary)
model = model.VAEModel(args.model, ntype_emb, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
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

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    kl_loss = 0
    ce_loss = 0

    with torch.no_grad():
      for i in range(0, len(data_source), args.bptt):
        data_full = data_source[i]
        data = data_full[:,0:data_full.size(1)-1]
        targets = data_full[:, 1:]
        #hidden = model.init_hidden(data.size(0))
        hidden = None
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        recon_batch, mu, log_var = model(data, None)
        kl,ce = loss_fn(recon_batch, targets,mu,log_var)
        loss = kl + ce
        total_loss += loss.item()
        kl_loss += kl
        ce_loss += ce
        #hidden = repackage_hidden(hidden)
    print("KL ", kl_loss, "CE: ", ce_loss)
    return kl_loss / i, ce_loss / i

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    kl_loss = 0
    ce_loss = 0
    for i in range(0, len(corpus.train), args.bptt):
        data_full = corpus.train[i]
        data_type  = corpus.train_type[i]
        print data_type.size(), data_full.size()
        data = data_full[:,0:data_full.size(1)-1]
        targets = data_full[:, 1:]
        #print data.size(), targets.size()
        hidden = None
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        #hidden = repackage_hidden(hidden)
        model.zero_grad()
        recon_batch, mu, log_var = model(data, None, data_full)
        #print output.size(), targets.size()
        kl,ce = loss_fn(recon_batch, targets,mu,log_var)
        loss  = kl + ce
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        kl_loss += kl.item()
        ce_loss += ce.item()
        total_loss += kl + ce

    return kl_loss/i , ce_loss/i



# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_klloss, train_celoss = train()
        #print("in train and val")
        dev_klloss,dev_celoss = evaluate(corpus.valid)
        val_loss = dev_klloss+dev_celoss
        print('-' * 89)
        #print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                   val_loss, math.exp(val_loss)))
        print("Epoch ", epoch, " Train KL Loss: ", train_klloss, " Train CE Loss: ", train_celoss, " Dev KL loss ", dev_klloss, " Dev CE Loss: ", dev_celoss)
        #print("end epoch, loss, ppl", epoch, val_loss.item()) #, math.exp(val_loss.item()))
        print('-' * 89)
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
test_klloss, test_celoss = evaluate(corpus.test)
test_loss = test_klloss + test_celoss
print('=' * 89)
print("Test KL Loss: ", test_klloss, "Test_celoss: ", test_celoss)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
