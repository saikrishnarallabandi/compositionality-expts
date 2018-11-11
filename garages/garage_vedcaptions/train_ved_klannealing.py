import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('/home/sirisha.rallabandi/projects/multimodal/repos/image_captioning')
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print_flag = 0

with open('../../features/captions.pkl', 'rb') as f:
    captions = pickle.load(f)

features = np.load('../../features/features.npy')

logfile_name = 'ved_klannealing_log'

g = open(logfile_name,'w')
g.close()

#### RESUME FLAG ####
resume = False
generation = False
model_name = './models/best-model.pkl'
####################

#sys.path.append('/home/ubuntu/captions/')

#from data_loader_barebones import get_loader
from model_ved import EncoderCNN, DecoderRNN, VED
from utilities import save_checkpoint, save_val_checkpoint, save_epoch, early_stopping, word_list, clean_sentence

start_time = time.time()

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 10          # number of training epochs
encoder_output_dim = 256 # dimensions of var and mu


# The size of the vocabulary
#vocab_size = len(train_loader.dataset.vocab)
vocab_size = 8855

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size, batch_size, encoder_output_dim)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
ved_model = VED(encoder, decoder)


# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    ved_model.cuda()

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=0).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(ignore_index=0)

# Specify the learnable parameters of the model
#params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
params = params + list(encoder.fc1.parameters()) + list(encoder.fc2.parameters()) + list(encoder.bottle_neck.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=0.001)

# Set the total number of training and validation steps per epoch

def kl_anneal_function(step, k=0.0025, x0=2500, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

class dataset(Dataset):

   def __init__(self, features, captions):
      self.features = features
      self.captions = captions

   def __len__(self):
        return len(self.features) 

   def __getitem__(self, item):
        return self.features[item], self.captions[item]  

trainset = dataset(features, captions)
train_loader = DataLoader(trainset, batch_size=1, shuffle=True)

curr_step = 0

def train():
  global curr_step
  total_loss = 0
  start_train_time = time.time()

  kl_loss = 0
  ce_loss = 0
  for i, (data) in enumerate(train_loader):
        curr_step += 1
        images, captions = data[0], data[1]
        if images.shape[1] == 1:
            continue
        if print_flag:
           print("Shape of images ", images.shape, " and that of captions: ", captions.shape)
        #images = images[0]
        #print("I got a batch of ", len(images), " images, ", captions.shape, " captions ")
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        #print("Shape of images: ", images.shape)
        #sys.exit()
        # Pass the inputs through the CNN-RNN model
        outputs, mu, logvar, z = ved_model(images, captions)

        # Calculate the batch loss
        weight = kl_anneal_function(curr_step)
        KLD = weight * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        ce = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        loss = ce + KLD
        kl_loss += KLD.item()
        ce_loss = ce.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
     
        if i % 1000 == 0:
            g = open(logfile_name,'a')
            g.write( " After processing " + str(i) + ' batches in time ' +  str(time.time() - start_train_time)  + ' seconds, train KL loss: ' + str(kl_loss) + ' train reconstruction loss: ' + str(ce_loss) + ' Perplexity: ' + str(np.exp(loss.item())) + '\n')
            g.close()




for epoch in range(10):
    train()

