import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import logging
import torch.nn.functional as F


#logging.basicConfig(level=logging.DEBUG)

class VAEModel(nn.Module):

    def __init__(self,rnn_type,ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
       super(VAEModel, self).__init__()
       self.embedding = nn.Embedding(ntoken, ninp)
       self.nlatent = 8
       self.fc1 = nn.Linear(ninp,nhid)
       self.fc2_a = nn.Linear(nhid, self.nlatent)
       self.fc2_b = nn.Linear(nhid, self.nlatent)       
       self.fc3 = nn.Linear(self.nlatent,int(self.nlatent*2))
       self.fc4 = nn.Linear(int(self.nlatent*2), ntoken)
       self.nlayers = nlayers
       self.nhid = nhid
       self.rnn_type = rnn_type
       self.rnn = nn.LSTM(nhid, int(nhid/2), num_layers=2, bidirectional=True)

    def encoder(self, emb, hidden):
       logging.debug("In Encoder")
       output, hidden = self.rnn(emb, hidden)
       
       logging.debug("Shape of output: {}".format(output.shape))

       h1 = F.relu(self.fc1(output))
       return self.fc2_a(h1), self.fc2_b(h1)

    def reparameterize(self, mu, log_var):
       std = torch.exp(0.5*log_var)
       eps = torch.rand_like(std)
       return eps.mul(std).add_(mu) 

    def forward(self,x,hidden):
       logging.debug("Shape of input to VAELM is {}".format(x.shape)) # [35, 20]
       embedding = self.embedding(x) # [35, 20, 200]
       logging.debug("Shape of embedding: {}".format(embedding.shape))
       mu, log_var = self.encoder(embedding,hidden)
       logging.debug("Shape of mu after encoder: {}".format(mu.shape))

       z = self.reparameterize(mu, log_var)
       logging.debug("Shape of latent representation: {}".format(z.shape))
       decoded = self.decode(z)
       logging.debug("Shape of decoder output: {}".format(decoded.shape))       
       return decoded,mu,log_var

    def decode(self,z):
       h3 = F.relu(self.fc3(z))
       return torch.sigmoid(self.fc4(h3))

        
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        temp1 = torch.nn.Parameter(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
        temp2 = torch.nn.Parameter(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
        temp = (temp1, temp2)
        logging.debug("I have initialized hidden as {}".format(temp1.shape))
        if self.rnn_type == 'LSTM':
            return temp



class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #logging.debug("Shape of input: {}".format(input.shape))
        #logging.debug("Shape of hidden[0]: {}".format(hidden[0].shape))
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        logging.debug("Shape of output after RNN: {}".format(output.shape))
        logging.debug("Shape of hidden[0]: {}".format(hidden[0].shape))
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        logging.debug("Shape of decoder output: {}".format(decoded.shape))
        #sys.exit()
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        #print(type(weight))
        temp1 = torch.nn.Parameter(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
        #temp = (Variable(torch.LongTensor(self.nlayers, bsz, self.nhid).fill_(0)).cuda(), Variable(torch.LongTensor(self.nlayers, bsz, self.nhid).fill_(0)).cuda())
        temp2 = torch.nn.Parameter(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
        temp = (temp1, temp2)
        #print (type(temp1))
        if self.rnn_type == 'LSTM':
            return temp

