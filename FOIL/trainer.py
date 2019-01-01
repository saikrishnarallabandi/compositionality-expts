import argparse
import torch
import torch.nn as nn
import numpy as np
import os, sys
import pickle
import time
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import RNN_model
from torch.nn.utils.rnn import pack_padded_sequence


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mask(lengths):
    """ Generate Mask """
    
    targets = torch.zeros(len(lengths), max(lengths)).long()
    for i, cap in enumerate(lengths):
        end = lengths[i]
        targets[i, :end] = 1

    return targets


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load foil data
    with open(args.foil_pickle_file, 'rb') as f:
        foil_data = pickle.load(f)

    # Build data loader
    data_loader = get_loader(foil_dict=foil_data,
                             vocab=vocab,
                             transform=None,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

    # Build the models
    num_outputs = 1
    decoder = RNN_model(args.feature_size, args.embed_size, args.hidden_size, len(vocab), num_outputs, args.num_layers).to(device)
    
    # Loss and optimizer
    loss_function = nn.BCEWithLogitsLoss().to(device)
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    save_num = 0
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        for i, (features, captions, lengths, targets, image_names) in enumerate(data_loader):
            save_num = save_num + 1
            # Set mini-batch dataset
            features = features.to(device)
            captions = captions.to(device)
            targets = targets.to(device)

            # Forward, backward and optimize
            outputs = decoder(features, captions, lengths)
        
            loss = loss_function(outputs.view(-1), targets.view(-1).float())
            #loss = criterion(outputs, targets)
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item())) 
                
            # Save the model checkpoints
            if (save_num) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        print('Epoch [{}/{}] time: {}'.format(epoch, args.num_epochs, time.time()-epoch_start_time)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--foil_pickle_file', type=str, default='./foil_train_with_features.pkl', help='path for foil training data file')
    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--feature_size', type=int , default=2048, help='dimension of image vectors')   
 
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
