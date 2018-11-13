import numpy as np
import os, sys
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from util import *
import logging

print_flag = 0
#logging.basicConfig(level=logging.DEBUG)

class custom_conv(nn.Conv1d):

    # Extending the default conv layer since we want to teacher force the input both during training and testing
    def __init__(self, *args, **kwargs):
        super(custom_conv, self).__init__(*args, **kwargs)
        self.input_buffer = None

    def incremental_forward(self, x):

        logging.debug("Shape of input in the custom conv increment forward function: {}".format(x.size()))
        x = x.data
        batch_size=  x.shape[0]
        kernel_size = self.kernel_size[0]
        logging.debug("Shape of weights in the custom conv layer: {}".format(self.weight.size())) # [num_outchannels, num_inchannels, kernel_size]

        # https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/conv.py, fairseq.modules.conv_tbc.ConvTBC
        weights = self.get_linearized_weights()
        logging.debug("Shape of linearized weights : {}".format(weights.size()))

        if self.input_buffer is None:
            self.input_buffer = x.new(batch_size, kernel_size, x.size(2))
            self.input_buffer.zero_()
            logging.debug(" Initialized an input buffer since none found")
        else:
            # Shift the existing buffer
            self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:,:].clone()
        self.input_buffer[:, -1,:] = x[:,-1,:]
        x = self.input_buffer

        logging.debug("Shape of input in the custom conv increment forward function after reshaping : {}".format(x.size()))
        logging.debug("Shape of weights in the custom conv increment forward function: {}".format(weights.size()))
        x_output = F.linear(x.view(batch_size, -1), weights)
        logging.debug("Finished incremental forward. returning a tensor of shape: {}".format(x_output.shape))
        return x_output.view(batch_size, 1, -1)
 

    def get_linearized_weights(self):
        weights = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
        weights = weights.view(self.out_channels, -1)
        return weights

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.contiguous().view(batch_size * time_steps, -1)
        x = self.module(x)
        x = x.contiguous().view(batch_size, time_steps, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class Model_MARC(torch.nn.Module):

     def __init__(self):
        super(Model_MARC, self).__init__()

        self.encoder = nn.Embedding(257, 128)
        self.encoder_fc = nn.Linear(128, 64)
        self.encoder_dropout = nn.Dropout(0.3)

        self.expert_conv = custom_conv(64, 128, kernel_size=5, stride=1, padding=2)
        self.gate_conv = custom_conv(64, 128, kernel_size=5, stride=1, padding=2)

        self.decoder_fc = SequenceWise(nn.Linear(128, 257))

     def forward(self,x):
        logging.debug("Shape of input the model: {}".format(x.shape)) 

        ## Encoder
        x = self.encoder(x)
        x = F.relu(self.encoder_fc(x))
        x = self.encoder_dropout(x)
        logging.debug("Shape of output from encoder: {}".format(x.shape))

        ## Our expert
        x = x.transpose(1,2) # Transpose the input for conv
        x_expert = self.expert_conv(x)
        x_gate = self.gate_conv(x)
        x_output = x_expert  * F.sigmoid(x_gate)
        x_output = x_output.transpose(1,2)
        logging.debug("Shape of output from gated cnn in forward: {}".format(x_output.shape))

        ## Decoder
        x = self.decoder_fc(x_output)
        return x


     def forward_sample(self,x):
        self.expert_conv.input_buffer = None
        self.gate_conv.input_buffer = None
        x = x.data
        max_len = x.shape[-1]
        logging.debug("Shape of input the model: {}".format(x.shape)) 
        x = self.encoder(x)
        x = F.relu(self.encoder_fc(x))
        x = self.encoder_dropout(x)
        logging.debug("Shape of output from encoder: {}".format(x.shape))

        # Transpose the input for CNN
        #x.transpose_(1,2)
        logging.debug("Shape of input being sent to incremental forward function: {}".format(x.shape))
       
        y = x.new(x.shape[0], max_len, 128)
        # Start loop
        for i in range(max_len):
           x_expert = self.expert_conv.incremental_forward(x) 
           x_gate = self.gate_conv.incremental_forward(x) 
           #logging.debug("Shape of output from expert: {}".format(x_expert.shape))
           #logging.debug("Shape of output from gate: {}".format(x_gate.shape))
           x_output = x_expert  * F.sigmoid(x_gate)
           logging.debug("Shape of output from after gating: {}".format(x_output.shape))
           logging.debug("Shape of y to which I am appending: {}".format(y.shape))
           y[:,i,:] = x_output[:,0,:].data
        #x_output.transpose_(1,2)
        logging.debug("Shape of output from gated cnn in forward_sample: {}".format(x_output.shape))

        ## Decoder
        y_output = self.decoder_fc(y)
        logging.debug(" ########## Decoding done ####### ")
        logging.debug("Shape of output from decoder that I am returing from the model: {}".format(y_output.shape))
 
        return y_output

