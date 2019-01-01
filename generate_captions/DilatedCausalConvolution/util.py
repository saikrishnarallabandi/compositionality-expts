import torch
import numpy as np
import torch.nn as nn


def sample_gumbel(shape, eps=1e-10, out=None):
   U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
   return - torch.log(eps - torch.log(U + eps))

def gumbel_argmax(logits, dim):
   # Draw from a multinomial distribution efficiently
   #print("Shape of gumbel input: ", logits.shape)
   return logits + sample_gumbel(logits.size(), out=logits.data.new())
   #sys.exit()
   return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


