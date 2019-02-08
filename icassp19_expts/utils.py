import pickle
import torch
import uuid
from data_loader import get_loader 
from build_vocab import Vocabulary
import numpy as np


def get_mask(lengths):
    """ Generate Mask """
    
    targets = torch.zeros(len(lengths), max(lengths)).long()
    for i, cap in enumerate(lengths):
        end = lengths[i]
        targets[i, :end] = 1

    return targets


# https://coderforevers.com/python/python-program/random-alphanumeric-string/
def get_random_string(string_length=5):
    """Returns a random string of length string_length."""

    # Convert UUID format to a Python string.
    random = str(uuid.uuid4())

    # Make all characters uppercase.
    random = random.upper()

    # Remove the UUID '-'.
    random = random.replace("-","")

    # Return the random string.
    return random[0:string_length]


def load_stuff(args):
   ### Load some stuff 
   with open(args.vocab_path, 'rb') as f:
      vocab = pickle.load(f)

   with open(args.imgid2caption_pickle_file, 'rb') as f:
      imageid2captions = pickle.load(f)

   with open(args.imgid2feature_pickle_file, 'rb') as f:
      imageid2features = pickle.load(f)

   with open(args.imgid2caption_pickle_file_val, 'rb') as f:
      imageid2captions_val = pickle.load(f)

   with open(args.imgid2feature_pickle_file_val, 'rb') as f:
      imageid2features_val = pickle.load(f)


   train_loader = get_loader(i2f_dict=imageid2features,
                             i2c_dict=imageid2captions,
                             vocab=vocab,
                             transform=None,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

   val_loader = get_loader(i2f_dict=imageid2features_val,
                             i2c_dict=imageid2captions_val,
                             vocab=vocab,
                             transform=None,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_workers)


   return train_loader, val_loader, vocab


def kl_anneal_function(step, k=0.0050, x0=2500, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

