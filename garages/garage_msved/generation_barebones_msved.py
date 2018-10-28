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

from utils_barebones_msved import *
from data_loader_barebones import get_loader
from model_barebones_msved import VED, EncoderCNN, DecoderRNN

start_time = time.time()

# Set values for the training variables
batch_size = 1         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
model_name = 'model_expA_msved.pth'


with open('train_loader.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader.pkl', 'rb') as handle:
    val_loader = pickle.load(handle)

print("Loading using pkl files took ", time.time() - start_time)


transform_test = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create the data loader
data_loader = get_loader(transform=transform_test,    
                         mode='test')





# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)

# Initialize the encoder and decoder
latent_spec = {'cont': 128, 'disc': [128]}
encoder = EncoderCNN(embed_size, batch_size, embed_size, latent_spec)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Initialize the VED model
with open(model_name, 'rb') as f:
        ved_model = torch.load(f)
ved_model.eval()

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    ved_model.cuda()

total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
print ("Number of validation steps:", total_val_step)

start_time = time.time()

get_prediction(data_loader, ved_model,train_loader.dataset.vocab)

