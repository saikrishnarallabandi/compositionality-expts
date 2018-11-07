import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import sys
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time
import pickle
from data_loader_barebones import get_loader

sys.path.append('/home/ubuntu/captions/')

start_time = time.time()

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5
vocab_from_file= True

transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Define a transform to pre-process the validation images
transform_val = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
train_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

val_loader = get_loader(transform=transform_val,
                         mode='val',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

#for i, data in enumerate(val_loader):
#   print(data)
#   continue


# https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
with open('train_loader_captions2014_batchsize32.pkl', 'wb') as handle:
    pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('valid_loader_caption2014_batchsize32.pkl', 'wb') as handle:
    pickle.dump(val_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print("Loading conventionally took ", time.time() - start_time)

#for i, data in enumerate(train_loader):
#   print(data)
#   continue

#sys.exit()

'''

with open('train_loader.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader.pkl', 'rb') as handle:
    val_loader = pickle.load(handle)

print("Loading using pkl files took ", time.time() - start_time)

'''

