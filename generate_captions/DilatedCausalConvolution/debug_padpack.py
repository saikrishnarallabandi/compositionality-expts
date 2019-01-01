import torch
from torch.nn.utils.rnn import pad_sequence


a = torch.ones(1,3)
b = torch.ones(1,4)
c = torch.ones(1,3)

padded_sequence = pad_sequence([a, b, c], batch_first=True)
print("Shapes of individual")
print("Shape of padded sequence is ", padded_sequence.shape)
