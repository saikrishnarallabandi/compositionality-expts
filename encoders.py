import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

sys.path.append('/home/srallaba/development/repos/falkon/')
import src.nn.layers as layers


class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()


#https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
#https://discuss.pytorch.org/t/custom-binarization-layer-with-straight-through-estimator-gives-error/4539/5
class straight_through(torch.autograd.Function):

     @staticmethod
     def forward(ctx, input):
         #print("Shape of input to the quantizer: ", input.shape)
         ctx.save_for_backward(input)
         #print("Shape of output from the quantizer: ", out.shape)
         return input

     @staticmethod
     def backward(ctx, grad_output):
         input, = ctx.saved_tensors
         grad_output[input>1]=0
         grad_output[input<-1]=0
         return grad_output

class quantizer(baseline_model):

        def __init__(self, num_classes, dimensions):
          super(quantizer, self).__init__()

          self.embedding = nn.Parameter(torch.rand(num_classes,dimensions))
          self.activation = straight_through.apply

        def forward(self, encoded):

          bsz = encoded.shape[0]
          T = encoded.shape[1]
          dims = encoded.shape[2]
          print("Shape of input to the quantizer: ", encoded.shape, " and that of the quantizer embedding: ", self.embedding.shape)
          sys.exit()
          encoded = encoded.reshape(bsz*T, dims)
          ## Loop over batch. (Cant you code better?)
          index_batch = []
          for chunk in encoded:
             c = (chunk - self.embedding).norm(dim=1)
             index_batch.append(torch.argmin(c))
          index_batch = torch.stack(index_batch, dim=0)  
           #self.activation(index_batch)
          quantized_values = torch.stack([self.embedding[k] for k in index_batch], dim=0)
          activated_values =  self.activation(quantized_values)
          return activated_values.reshape(bsz, T, dims)
          #return self.activation(quantized_values)


