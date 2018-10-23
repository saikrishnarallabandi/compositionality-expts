import os,sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from itertools import dropwhile
import nltk
import string
from nltk import word_tokenize

wids_global = defaultdict(lambda: len(wids_global))

class vqa_dataset(Dataset):

   # Dataset for utterances and types
    def __init__(self, file, train_flag, wids=None):
        self.utts = []
        self.types = []
        if train_flag:
          wids = wids_global
        wids['_PAD'] = 0
        wids['<sos>'] = 1
        wids['<eos>'] = 2
        wids['UNK'] = 3
        word_tokens = self.remove_rarewords(file, 10)
        #sys.exit()
        f = open(file)
        c = 0
        for line in f:
           c+=1
           if c > 10:
               #For debugging, faster to load just 10 lines
               continue
           line = word_tokenize(line.split('\n')[0])
           for i, w in enumerate(line):
               if not train_flag: # Validation Mode / Testing Mode
                 if w in wids and w in word_tokens:
                     pass
                 else:
                     line[i] = 'UNK'
               elif train_flag and w in word_tokens: # Training Mode and not rare word
                     wid = wids[w]
               else: # Training mode but rare word
                    line[i] = 'UNK'
           self.utts.append([1]  + [wids[w] for w in line] + [2])
           self.types.append(self.get_type(line))

    def remove_rarewords(self, file, freq):
       words = defaultdict(int)
       freq = int(freq)
       c = 0
       f = open(file)
       for line in f:
           c += 1
           if c > 10: # For debugging, faster to load just 10 lines
               continue
           line =  line.split('\n')[0]
           line = word_tokenize(line) + ['_PAD'] + ['<sos>'] + ['<eos>'] + ["UNK"] # Punctuation and stuff
           #print(line)
           for w in line:
              words[w] += 1
       for k in list(words):
         #print(k, words[k])
         if words[k] < freq:
            ##print("Deleting")
            ##print('\n')
            del words[k]
       f.close()
       return words

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, item):
        return self.utts[item], self.types[item]

    def get_wids():
        return wids_global

    def get_type(self,line):
      if line[0] == "Is" or line[0] == 'Are':
           return 0
      elif line[0] == "How" and line[1] == "many":
           return 1
      else:
           return 2


def collate_fn(batch):
    """Create batch"""

    #print(batch)
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths) + 1

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[0]) for x in batch]) + 1

    a = np.array( [ _pad(x[0], max_input_len)  for x in batch ], dtype=np.int)
    b = np.array( [ x[1] for x in batch ], dtype=np.int)
    a_batch = torch.LongTensor(a)
    b_batch = torch.LongTensor(b)
    input_lengths = torch.LongTensor(input_lengths)

    return a_batch, b_batch


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)
