import torch
import numpy as np 
import argparse
import pickle 
import os
from build_vocab import Vocabulary
from model import *


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load image id 2 caption pickle file
    with open(args.imgid2caption_pickle_file, 'rb') as f:
        imageid2captions = pickle.load(f)

    # Load image id 2 feature pickle file
    with open(args.imgid2feature_pickle_file, 'rb') as f:
        imageid2features = pickle.load(f)

    # Build models
    decoder = CaptionSingleCNN(args.feature_size, args.embed_size, args.hidden_size, len(vocab), args.num_layers).eval() ## Batch norm
    decoder = decoder.to(device)

    # Load the trained model parameters
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Ids
    ids = imageid2features.keys()
    
    for i, id in enumerate(ids):
        # Prepare features
        feature = torch.Tensor(imageid2features[id]).to(device)
        #caption = torch.Tensor(imageid2captions[id]).to(device)
        feature = feature.unsqueeze(0)
        #caption = caption.unsqueeze(0)
    
        # Generate an caption from the image
        sampled_ids = decoder.sample(feature)
        #sampled_ids = decoder(feature, caption)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        #sampled_ids = [torch.max(k)[1] for k in sampled_ids]
        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
    
        # Print out the image and the generated caption
        print (sentence)
        print("Original: ", imageid2captions[id]) 
        print('\n')
        
        if i > 100:
            sys.exit()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_path', type=str, default='./models_cnn/decoder-4-700.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--imgid2caption_pickle_file', type=str, default='./imageid2captions_val.pkl', help='path for image 2 captions pickle file')
    parser.add_argument('--imgid2feature_pickle_file', type=str, default='./imageid2features_val.pkl', help='path for image 2 features pickle file')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--feature_size', type=int , default=2048, help='dimension of image vectors')
    args = parser.parse_args()
    main(args)
