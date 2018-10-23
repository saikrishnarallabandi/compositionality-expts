import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

wids = defaultdict(lambda: len(wids))

class vqa_dataset(Dataset):

   # Dataset for utterances and types
    def __init__(self, file):
        f = open(file)
        self.utts = []
        self.types = []
        wids['_PAD'] = 0
        wids['<sos>'] = 1
        wids['<eos>'] = 2
        print(wids['<eos>'])
        for line in f:
           line = line.split('\n')[0].split()
           for w in line:
               wid = wids[w]
               #print(wid)
           self.utts.append([1]  + [wids[w] for w in line] + [2])
           #print([1]  + [self.wids[w] for w in line] + [2]) 
           if line[0] == "Is" or line[0] == 'Are':
              self.types.append(0)
           elif line[0] == "How many":
              self.types.append(1)
           else:
              self.types.append(2)

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, item):
        return self.utts[item], self.types[item]

    def get_wids():
        return wids

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


