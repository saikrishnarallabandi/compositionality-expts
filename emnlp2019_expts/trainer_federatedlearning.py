import argparse
import torch
import torch.nn as nn
import numpy as np
import os, sys
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import *
from torch.nn.utils.rnn import pack_padded_sequence
from logger import *
from utils import *

### Some stuff
updates = 0
log_flag = 1
exp_name = 'exp_rnnbaseline_nopadpack_' + str(get_random_string())
exp_dir = 'exp/' + exp_name
if not os.path.exists(exp_dir):
   os.mkdir(exp_dir)
   os.mkdir(exp_dir + '/models')
   os.mkdir(exp_dir + '/logs')
logger = Logger(exp_dir + '/logs/' + exp_name)
model_dir = exp_dir + '/models'

falcon_dir = '/home/srallaba/projects/caption_generation/repos/falkon'
sys.path.append(falcon_dir)
import src.nn.layers as layers

### Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def modify_loss(loss):
   
   #print("Shape of the loss: ", loss.shape)
   loss_softmax = F.softmax(loss)
   #print("Softmax ed loss is ", loss_softmax.shape)
   loss = loss * loss_softmax
   #print("Softmax weighted loss is ", loss.shape)
   return torch.mean(loss, dim=0)
   exit() 
  
   probs = []
   masks = []
   for l in loss:
       mask = 0
       prob = l.detach().item()
       probs.append(prob)
       choice = np.random.rand()
       choice = 0.4
       if prob < 0.2 : #&& choice > prob:
          mask = 0
       masks.append(mask)       
   #print("Backprop probabilities and masks are : ", probs, masks)
   #print("Modified loss vector is ", loss)
   loss = torch.mean(mask * loss, dim=0)
   #print("Modified loss is ", loss)      
   return loss  

def val():

    model.eval()
    total_loss = 0
    klloss = 0
    with torch.no_grad():
       for i, (features, captions, lengths, image_names) in enumerate(val_loader):

            features = features.to(device)
            captions = captions.numpy().squeeze(0)
            outputs = model.sample(features).squeeze(0)
            outputs = outputs.cpu().numpy().squeeze(-1)             
            print(image_names[0], outputs.shape)
            print(' Original Caption: ' + ' '.join(vocab.idx2word[k] for k in captions) )
            print(' Predicted Caption: ' + ' '.join(vocab.idx2word[k] for k in outputs))
            print('\n')
            if i == 1:
               return 0
        

def train():

    model.train()
    global updates
    total_loss = 0
    total_step = len(train_loader)
    for i, (features, captions, lengths, image_names) in enumerate(train_loader):

            updates += 1

            features = features.to(device)
            captions = captions.to(device)
            #print("Shape of features and captions: ", features.shape, captions.shape)

            outputs = model(features, captions)
            #print("Shape of outputs and captions: ", outputs.shape, captions.shape, len(vocab))
            loss = criterion(outputs.contiguous().view(-1, len(vocab)), captions.view(-1))
            loss = modify_loss(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()
            if log_flag:
               logger.scalar_summary('Train Loss', loss.item(), updates)

            if updates % 100 == 1:
               print("After ", updates, " updates, training loss: ", total_loss/(i+1))



    return total_loss/(i)


def main(args):

   train_loader, val_loader, vocab = load_stuff(args)
   model = CaptionRNN_nopadpack(args.feature_size, args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
   criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
   params = list(model.parameters())
   optimizer = torch.optim.Adam(params, lr=args.learning_rate)
   global model, train_loader, val_loader, criterion, optimizer, vocab

   for epoch in range(args.num_epochs):

      train_loss = train()
      val_loss = val()
      val_loss = 0
      if log_flag:
            logger.scalar_summary('Train Loss per epoch', train_loss , epoch)
            logger.scalar_summary('Val Loss per epoch ', val_loss , epoch)

      # Save the model checkpoints
      torch.save(model.state_dict(), os.path.join(
                model_dir , 'model-{}-{}.ckpt'.format(epoch+1, updates+1)))
      print("Saved the model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--imgid2caption_pickle_file', type=str, default='./imageid2allcaptions_train.pkl', help='path for image 2 captions pickle file')
    parser.add_argument('--imgid2feature_pickle_file', type=str, default='./imageid2features.pkl', help='path for image 2 features pickle file')
    parser.add_argument('--imgid2caption_pickle_file_val', type=str, default='./imageid2allcaptions_val.pkl', help='path for image 2 captions pickle file')
    parser.add_argument('--imgid2feature_pickle_file_val', type=str, default='./imageid2features_val.pkl', help='path for image 2 features pickle file')

    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size_image', type=int , default=256, help='dimension of image vectors')
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--feature_size', type=int , default=2048, help='dimension of image vectors')   
    parser.add_argument('--clip', type=float , default=0.25, help='dimension of image vectors')  

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
