import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import sys
from pycocotools.coco import COCO
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print_flag = 0

sys.path.append('/home/ubuntu/captions/')

f = open('captions_generated.txt', 'w')
f.close()

#from utils_barebones import train, validate, save_epoch, early_stopping
#from data_loader_barebones import get_loader
from model_ved import EncoderCNN, DecoderRNN, VED
from utilities import save_checkpoint, save_val_checkpoint, save_epoch, early_stopping, word_list, clean_sentence

import torch.utils.data as data
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


#### RESUME FLAG ####
resume = False
generation = False
model_name = './models/best-model.pkl'
####################


start_time = time.time()

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 1          # number of training epochs
encoder_output_dim = 32

with open('captions.pkl', 'rb') as f:
    captions = pickle.load(f)

features = np.load('features.npy')

with open('train_loader_captions2014_batchsize32.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader_caption2014_batchsize32.pkl', 'rb') as handle:
    val_loader = pickle.load(handle)

print("Loading using pkl files took ", time.time() - start_time)

# The size of the vocabulary
vocab = train_loader.dataset.vocab
vocab_size = len(train_loader.dataset.vocab)
print("Vocab size is ", vocab_size)
# get_prediction(data_loader, ved_model,train_loader.dataset.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size, hidden_size, encoder_output_dim)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, encoder_output_dim)
ved_model = VED(encoder, decoder)

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=0).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(ignore_index=0)

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=0.001)

# Set the total number of training and validation steps per epoch
total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
print ("Number of training steps:", total_train_step)
print ("Number of validation steps:", total_val_step)

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
valset = dataset(features, captions)
val_loader = DataLoader(valset, batch_size=1, shuffle=False)


def eval():
  ved_model.eval()
  total_loss = 0
  with torch.no_grad():      
    for i, (data) in enumerate(train_loader):
        curr_step += 1
        features, captions = data[0], data[1]
        if print_flag:
           print("Shape of features is ", features.shape, " and the shape of captions is ", captions.shape)

        if torch.cuda.is_available():
            features = features.cuda()
            captions = captions.cuda()

        # Pass the inputs through the VED model
        outputs,mu,logvar,z = ved_model(features, captions)

        # Calculate the batch loss
        weight = kl_anneal_function(curr_step)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ce = criterion(outputs.view(-1, vocab_size), captions.view(-1)) 
        loss = ce + weight*KLD

        total_loss += loss.item()
        if i % 50 == 0:
            return total_loss / 50.0



def get_prediction(data_loader, ved_model, vocab):
    """Loop over images in a dataset and print model's top three predicted
    captions using beam search."""
    ved_model.eval()
    l = []
    for i, (features, captions) in enumerate(data_loader):
        if i > 0:
            continue
        if torch.cuda.is_available():
            features = features.cuda()
            captions = captions.cuda()

        features = features[:,0,:]
        captions = captions[:,0,:]
        if print_flag:
           print("Shape of features and captions going to the sampling function: ", features.shape)
        output = ved_model.sample(features)
        #if print_flag:
        #   print("I got this as the output from sampling function: ", output.shape)
        sentence = clean_sentence(output, vocab)
        orig_caption = clean_sentence(captions.detach().cpu().numpy(), vocab)
        print("   I predicted: ", ' '.join(k for k in sentence), " while the original was ", orig_caption)
        #path_parts = path[0].split(".")
        #l.append({'image_id':path_parts[0][-6:], 'caption':sentence})
    #with open('captions_generated.txt', 'a') as f:
    #    f.write(sentence + '\n')
    return
    exit(1)


def train():
  global curr_step
  total_loss = 0
  kl_loss = 0
  ce_loss = 0
  start_train_time = time.time()
  for i, (data) in enumerate(train_loader):
        curr_step += 1
        features, captions = data[0], data[1]
        if print_flag:
           print("Shape of features is ", features.shape, " and the shape of captions is ", captions.shape)

        if torch.cuda.is_available():
            features = features.cuda()
            captions = captions.cuda()

        # Pass the inputs through the VED model
        outputs,mu,logvar,z = ved_model(features, captions)

        # Calculate the batch loss
        weight = kl_anneal_function(curr_step)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ce = criterion(outputs.view(-1, vocab_size), captions.view(-1)) 
        kl_loss += KLD.item()
        ce_loss += ce.item()

        loss = ce + weight*KLD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 500 == 0:
            print("   After ", i, " steps train KL loss: ", kl_loss/(i+1), " train reconstruction loss: ", ce_loss/(i+1), " in ", time.time() - start_train_time, " seconds" )
            get_prediction(val_loader, ved_model, vocab)
            ved_model.train()

train()
