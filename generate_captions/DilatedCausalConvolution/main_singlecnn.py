import torch
import torch.nn as nn
import numpy as np
import os, sys
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import *
import time
from layers import *

## Flags
print_flag = 0
debug_flag = 0

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
    
print(vocab.idx2word[0])    

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
hidden_size = 128
model = CaptionSingleCNN(feature_size,embed_size, hidden_size,len(vocab),2).to(device)
print(model)
criterion = nn.CrossEntropyLoss(ignore_index=0,reduction='none')
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr = 0.001)
updates = 0

def loss_fn(outputs, targets):
    
    bsz = outputs.shape[0]
    #print("Shapes of outputs and targets in the loss function: ", outputs.shape, targets.shape)
    loss_value = criterion(outputs, targets)
    #print("Shape of loss values is ", loss_value.shape, bsz)
    #return loss_value
    return loss_value.mean()

def generate(features, captions):
    
    features = features.unsqueeze(0)
    captions = captions.unsqueeze(0)
    
    # Generate an caption from the image
    sampled_ids = model.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    #print("Shape of sampled ids during generation: ", sampled_ids.shape, sampled_ids)

    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        #print("Word Id is: ", word_id)
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
             break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print ("I predicted: ", sentence)
    
    sampled_ids = captions
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
             break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print ("Original Sentence: ", sentence)
    
    print('\n')
    
    return

    
## Validation
def val(partial_flag= 1):
  model.eval()
  l = 0
  with torch.no_grad():
    for i, (features, captions, lengths, image_names) in enumerate(val_loader):
                  
            features = features.to(device)
            captions = captions.to(device)

            
            features = features[0,:]
            captions = captions[0,:]
            print("Shape of features and captions: ", features.shape, captions.shape)
            generate(features, captions)
            #sys.exit()
            return l/(i+1)
     
    return l/(i+1)
    
## Train 
def train():
    optimizer.zero_grad()
    model.train()
    global updates 
    l = 0
    for i, (features, captions, lengths, image_names) in enumerate(train_loader):
            
            updates += 1
                        
            features = features.to(device)
            captions = captions.to(device)
            outputs = model(features, captions, lengths)
            bsz = features.shape[0]
            t = captions.shape[1]
            #print("Shape of outputs, captions", outputs.shape, captions.shape)
            #print("bsz,t: ", bsz,t)
            targets = captions.contiguous().view(bsz*t)
            outputs = outputs.contiguous().view(bsz*t,-1)

            #print("Shape of outputs and targets: ", outputs.shape, targets.shape)
            loss = loss_fn(outputs,targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.25)
            optimizer.step()
            
            l += loss.item()
            
            if i % 100 == 1:
                print("  After ", i, " batches train loss: ", l/(i+1), loss.item())
                

     
    return l/(i+1)
    
    
    
## Main Loop
for epoch in range(10):
    epoch_start_time = time.time()
    train_loss = train()
    val_loss = val(0)
    g = open(log_file, 'a')
    g.write("Epoch: " +  str(epoch).zfill(3) + " Train Loss: " +  str(train_loss) +  " Val Loss: " +  str(val_loss)  +  " Time per epoch: " + str(time.time() - epoch_start_time) + " seconds" + '\n')    
    g.close()   
 
