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


sys.path.append('/home/ubuntu/captions/')

#from utils_barebones import train, validate, save_epoch, early_stopping
from data_loader_barebones import get_loader
from model_rnn import EncoderCNN, DecoderRNN
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
num_epochs = 25          # number of training epochs

with open('captions.pkl', 'rb') as f:
    captions = pickle.load(f)

features = np.load('features.npy')

with open('train_loader_captions2014_batchsize32.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader_caption2014_batchsize32.pkl', 'rb') as handle:
    val_loader = pickle.load(handle)

with open('captions_val.pkl', 'rb') as f:
    captions_val = pickle.load(f)

features_val = np.load('features_val.npy')



print("Loading using pkl files took ", time.time() - start_time)

# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)
print("Vocab size is ", vocab_size)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

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


class dataset(Dataset):

   def __init__(self, features, captions,mode):
      self.mode = mode 
      self.features = features
      self.captions = captions

   def __len__(self):
        return len(self.features) 

   def __getitem__(self, item):
        return self.features[item], self.captions[item]  

trainset = dataset(features, captions)
train_loader = DataLoader(trainset, batch_size=1, shuffle=True)

total_loss = 0
start_train_time = time.time()

for i, (data) in enumerate(train_loader):

        images, captions = data[0], data[1]
        #images = images[0]
        #print("I got a batch of ", len(images), " images, ", captions.shape, " captions ")
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        #print("Shape of images: ", images.shape)
        #sys.exit()
        # Pass the inputs through the CNN-RNN model
        features = encoder.forward_features(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
        if i % 500 == 0:
            print(i, time.time() - start_train_time,total_loss, np.exp(loss.item()))


