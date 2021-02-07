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

sys.path.append('/home/ubuntu/captions/')

from utils_barebones_msved import train, validate, save_epoch, early_stopping
from data_loader_barebones import get_loader
from model_barebones_msved import VED, EncoderCNN, DecoderRNN

start_time = time.time()

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 10          # number of training epochs



with open('train_loader.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader.pkl', 'rb') as handle:
    val_loader = pickle.load(handle)

print("Loading using pkl files took ", time.time() - start_time)

# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)

# Initialize the encoder and decoder
latent_spec = {'cont': 128, 'disc': [128]}
encoder = EncoderCNN(embed_size, batch_size, embed_size, latent_spec)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Initialize the VED model
ved_model = VED(encoder, decoder)

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    ved_model.cuda()

# Define the loss function
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model
params = list(ved_model.decoder.parameters()) + list(ved_model.encoder.features_to_hidden.parameters()) + list(ved_model.encoder.fc_mean.parameters()) + list(ved_model.encoder.fc_log_var.parameters()) + list(ved_model.encoder.fc_alphas.parameters()) 
#params = params + list(ved_model.encoder.fc1.parameters()) + list(ved_model.encoder.fc2.parameters())

# Define the optimizer 
optimizer = torch.optim.Adam(params=params, lr=0.001)

# Set the total number of training and validation steps per epoch
total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
print ("Number of training steps:", total_train_step)
print ("Number of validation steps:", total_val_step)


# Keep track of train and validation losses and validation Bleu-4 scores by epoch
train_losses = []
val_losses = []
val_bleus = []
# Keep track of the current best validation Bleu score
best_val_bleu = float("-INF")

start_time = time.time()
num_epochs = 4
for epoch in range(1, num_epochs + 1):
    train_loss = train(train_loader, ved_model, criterion, optimizer, vocab_size, epoch, total_train_step)
    
    val_loss, val_bleu = validate(val_loader, ved_model, criterion,
                                  train_loader.dataset.vocab, epoch, total_val_step)

    print("---------------------After Epoch : {}------------------".format(epoch)) 
    print("Train Loss : ", train_loss)
    print("Valid Loss : ", val_loss)
    print("Bleu Score: ", val_bleu)
    print("-------------------------------------------------------")
