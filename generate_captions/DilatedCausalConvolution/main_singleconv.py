import torch
import torch.nn as nn
import numpy as np
import os, sys
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import *
import time

## Flags
print_flag = 0
debug_flag = 1

## Files
vocab_file = 'vocab.pkl'
imageid2captions_train_file = 'imageid2captions.pkl'
imageid2features_train_file = 'imageid2features.pkl'
imageid2captions_val_file = 'imageid2captions_val.pkl'
imageid2features_val_file = 'imageid2features_val.pkl'
log_file = 'log_singleconv'
g = open(log_file,'w')
g.close()

## Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load the files
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

with open(imageid2captions_train_file, 'rb') as f:
    imageid2captions_train = pickle.load(f)

with open(imageid2features_train_file, 'rb') as f:
    imageid2features_train = pickle.load(f)

with open(imageid2captions_val_file, 'rb') as f:
    imageid2captions_val = pickle.load(f)

with open(imageid2features_val_file, 'rb') as f:
    imageid2features_val = pickle.load(f)


## Dataloaders
train_loader = get_loader(i2f_dict=imageid2features_train,
                             i2c_dict=imageid2captions_train,
                             vocab=vocab,
                             transform=None,
                             batch_size=32,
                             shuffle=True,
                             num_workers=4)

val_loader = get_loader(i2f_dict=imageid2features_val,
                             i2c_dict=imageid2captions_val,
                             vocab=vocab,
                             transform=None,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)

## Model and stuff
feature_size = 2048
embed_size = 256
hidden_size = 512
model = CaptionSingleCNN(feature_size,embed_size, hidden_size,len(vocab),1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr = 0.001)
updates = 0

## Validation
def val(partial_flag= 1):
    model.eval()
    l = 0
    for i, (features, captions, lengths, image_names) in enumerate(val_loader):
                  
            features = features.to(device)
            captions = captions.to(device)
            outputs = model.sample(features,return_logits=1)
            bsz = features.shape[0]
            outputs = outputs[:captions.shape[1],:,:]
            outputs = outputs.squeeze(1)
            #print("Shape of outputs and captions: ", outputs.shape, captions.shape)
            loss = criterion(outputs,captions.reshape(captions.shape[0]*captions.shape[1]))
            l += loss.item()
            
            if i == 1 and partial_flag:
               #print("  Val loop: After ", i, " batches, loss: ", l/(i+1))
               output = model.sample(features)
               sampled_ids = torch.max(outputs,dim=1)[1]
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                     break
               sentence = ' '.join(sampled_caption)
    
               # Print out the image and the generated caption
               print ("  Val mein Predicted: ", sentence)
               
               outputs = captions[0,:]
               sampled_ids = outputs
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                   break
               sentence = ' '.join(sampled_caption)
    
               # Print out the image and the generated caption
               print ("  Val mein Original: ", sentence)
               print('\n')
               return l/(i+1)
     
    return l/(i+1)
    
## Train 
def train():
    model.train()
    global updates 
    l = 0
    for i, (features, captions, lengths, image_names) in enumerate(train_loader):
            
            updates += 1
                        
            features = features.to(device)
            captions = captions.to(device)
            outputs = model(features, captions, lengths)
            bsz = features.shape[0]
            
            if debug_flag:
               captions_bkp = captions
               outputs_bkp = outputs

            loss = criterion(outputs.reshape(bsz*outputs.shape[1], outputs.shape[2]),captions.reshape(captions.shape[0] * captions.shape[1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            l += loss.item()
            
            if debug_flag and i% 300 == 1:
               captions = captions_bkp
               outputs = outputs_bkp
               captions = captions[0,:]
               outputs = outputs[0,:,:]
               sampled_ids = torch.max(outputs,dim=1)[1]
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                     break
               sentence = ' '.join(sampled_caption)
    
               # Print out the image and the generated caption
               print ("  Train mein Predicted: ", sentence)

               outputs = captions
               sampled_ids = outputs
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                   break
               sentence = ' '.join(sampled_caption)
    
               # Print out the image and the generated caption
               print ("  Train mein Original: ", sentence)
                

     
    return l/(i+1)
    
    
    
## Main Loop
for epoch in range(10):
    epoch_start_time = time.time()
    train_loss = train()
    val_loss = val()
    g = open(log_file, 'a')
    g.write("Epoch: " +  str(epoch).zfill(3) + " Train Loss: " +  str(train_loss) +  " Val Loss: " +  str(val_loss)  +  " Time per epoch: " + str(time.time() - epoch_start_time) + " seconds" + '\n')    
    g.close()   
 
