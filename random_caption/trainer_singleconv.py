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


debug_flag = 0

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

    # Load image id 2 caption pickle file
    with open(args.imgid2caption_pickle_file, 'rb') as f:
        imageid2captions = pickle.load(f)

    # Load image id 2 feature pickle file
    with open(args.imgid2feature_pickle_file, 'rb') as f:
        imageid2features = pickle.load(f)

    # Build data loader
    data_loader = get_loader(i2f_dict=imageid2features,
                             i2c_dict=imageid2captions,
                             vocab=vocab,
                             transform=None,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

    # Build the models
    decoder = CaptionSingleCNN(args.feature_size, args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        l = 0
        for i, (features, captions, lengths, image_names) in enumerate(data_loader):
            # Set mini-batch dataset
            features = features.to(device)
            captions = captions.to(device)
            #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            outputs = decoder(features, captions, lengths)
            bsz = features.shape[0]
            
            if i % 300 == 1 and debug_flag:
                outputs = outputs[0,:,:]
                print(outputs.shape)
                sampled_ids = torch.max(outputs,dim=1)[1]
                print(sampled_ids.shape)
                sampled_ids = sampled_ids.cpu().numpy()
                sampled_caption = []
                for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                   break
                sentence = ' '.join(sampled_caption)
    
                # Print out the image and the generated caption
                print (sentence)
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
                print ("Original: ", sentence)
                print('\n')
                
                continue
                
            #print("Shape of outputs and targets is ", outputs.shape, captions.shape) 
            loss = criterion(outputs.reshape(bsz*outputs.shape[1], outputs.shape[2]), captions.reshape(captions.shape[0] * captions.shape[1]))
            l += loss.item()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()


            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, l/(i+1), np.exp(l/(i+1)))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_cnn/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--imgid2caption_pickle_file', type=str, default='./imageid2captions.pkl', help='path for image 2 captions pickle file')
    parser.add_argument('--imgid2feature_pickle_file', type=str, default='./imageid2features.pkl', help='path for image 2 features pickle file')
    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=100, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--feature_size', type=int , default=2048, help='dimension of image vectors')   
 
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
