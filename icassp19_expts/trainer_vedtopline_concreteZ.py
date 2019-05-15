import argparse
import torch
import torch.nn as nn
import numpy as np
import os, sys
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model_concreteved import *
from torch.nn.utils.rnn import pack_padded_sequence
from logger import *
from utils import *
import time

### Some stuff
updates = 0
log_flag = 1
exp_name = 'exp_vedtopline_temporalz_128latents_0.0025weightrate_10kstep_sampling' + str(get_random_string())
exp_dir = 'exp/' + exp_name
if not os.path.exists(exp_dir):
   os.mkdir(exp_dir)
   os.mkdir(exp_dir + '/models')
   os.mkdir(exp_dir + '/logs')
logger = Logger(exp_dir + '/logs/' + exp_name)
model_dir = exp_dir + '/models'
save_model_flag = 0
falcon_dir = '/home/srallaba/projects/caption_generation/repos/falkon'
sys.path.append(falcon_dir)
import src.nn.layers as layers

### Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = 0
EPS=1e-12

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_continuous(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

def loss_function_discrete(alpha):
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if torch.cuda.is_available():
       log_dim = log_dim.cuda()
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    kl_loss = log_dim + mean_neg_entropy
    return torch.sum(kl_loss)

def val_spatial():

    model.eval()
    total_loss = 0
    klloss = 0
    with torch.no_grad():
       for i, (features, captions, lengths, image_names) in enumerate(val_loader):

            features = features.to(device)
            captions = captions.numpy().squeeze(0)
            outputs = model.sample_greedy_ConditionalZ_spatial(features)
            outputs = outputs.cpu().numpy().squeeze(-1)
            outputs = outputs.squeeze(0)  
            print(image_names[0])
            print(' Original Caption: ' + ' '.join(vocab.idx2word[k] for k in captions) )
            print(' Predicted Caption: ' + ' '.join(vocab.idx2word[k] for k in outputs))
            print('\n')
            if i == 1:
               return 0

def visualize_spatial():

    model.eval()
    total_loss = 0
    klloss = 0
    with torch.no_grad():
       for i, (features, captions, lengths, image_names) in enumerate(val_loader):

            features = features.to(device)
            captions = captions.numpy().squeeze(0)
            outputs = model.sample_greedy_ConditionalZ_spatial_visualize(features)
            outputs = outputs.cpu().numpy().squeeze(-1)
            outputs = outputs.squeeze(0)
            print(image_names[0])
            print(' Original Caption: ' + ' '.join(vocab.idx2word[k] for k in captions) )
            print(' Predicted Caption: ' + ' '.join(vocab.idx2word[k] for k in outputs))
            print('\n')
            if i == 1:
               return 0

def train_spatial():        

    start_time = time.time()
    model.train()
    global updates
    total_loss = 0
    klloss = 0
    weight = 0
    total_step = len(train_loader)
    for i, (features, captions, lengths, image_names) in enumerate(train_loader):

            updates += 1

            features = features.to(device)
            captions = captions.to(device)

            outputs, mu, logvar, alpha = model.forward_ConditionalZ_spatial(features, captions)
            captions = captions[:,1:]
           
            loss = criterion(outputs.contiguous().view(-1, len(vocab)), captions.contiguous().view(-1))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) + c
            kl_loss_discrete = loss_function_discrete(alpha)
            #print("Shapes of losses: ", loss.shape, loss, kl_loss.shape, kl_loss, kl_loss_discrete.shape, kl_loss_discrete)
            loss += kl_loss + kl_loss_discrete
            klloss += kl_loss.item() + kl_loss_discrete.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()
            if log_flag:
               logger.scalar_summary('Train Loss', loss.item(), updates)
               logger.scalar_summary('KL Loss', klloss/(i+1), updates)

            if updates % 100 == 1:
               print("After ", updates, " updates, training loss: ", total_loss/(i+1), " KL loss is ", klloss/(i+1), " weight: ", weight, " it took ", time.time() - start_time)

               captions = captions[0,:].detach().cpu().numpy()
               outputs = outputs.max(-1)[1]
               outputs = outputs[0,:].detach().cpu().numpy()

    return total_loss/(i)



def main(args):

   train_loader, val_loader, vocab = load_stuff(args)
   model = CaptionRNN_VItopline(args.feature_size, args.embed_size_image, args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
   criterion = nn.CrossEntropyLoss(ignore_index=0)
   params = list(model.parameters())
   optimizer = torch.optim.Adam(params, lr=args.learning_rate)
   global model, train_loader, val_loader, criterion, optimizer, vocab

   for epoch in range(args.num_epochs):

      train_loss = train_spatial()
      val_loss = val_spatial()
      #val_loss = 0
      if log_flag:
            logger.scalar_summary('Train Loss per epoch', train_loss , epoch)
            logger.scalar_summary('Val Loss per epoch ', val_loss , epoch)

      # Save the model checkpoints
      if save_model_flag:
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
    parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')
    parser.add_argument('--feature_size', type=int , default=2048, help='dimension of image vectors')   
    parser.add_argument('--clip', type=float , default=0.25, help='dimension of image vectors')  

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

