import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from model_utils import *

print_flag = 0



class AdvancedConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw + (kw - 1) * (dilation - 1), input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        if print_flag:
           print("Shape of input and the weight: ", input.shape, weight.shape)
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)


    def clear_buffer(self):
        self.input_buffer = None

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight



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
        x = x.contiguous()
        x = x.view(batch_size * time_steps, -1)
        x = self.module(x)
        x = x.contiguous()
        x = x.view(batch_size, time_steps, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MultimodalResidualConv1d(AdvancedConv1d):
      
      def __init__(self, input_channels, conditioning_channels, gate_channels, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.input_channels = input_channels
          self.conditioning_channels = conditioning_channels
          self.gate_channels = gate_channels

          self.input2gate_conv = AdvancedConv1d(self.input_channels,self.gate_channels*2, kernel_size=3,padding=1,stride=1)
          self.conditioning2gate_conv = AdvancedConv1d(self.conditioning_channels,self.gate_channels*2, kernel_size=3,padding=1,stride=1)
          self.combiner2input_conv = AdvancedConv1d(self.gate_channels, self.input_channels,kernel_size=3,padding=1,stride=1)

      def forward(self, x, c):
          return _forward(x, c, 0)

      def forward_incremental(self, x,c):
          return _forward(x, c, 1)

      def _forward(self, x, c, incremental_flag):

          # Convert input to gate
          if incremental_flag:
             x = self.input2gate_conv.incremental_forward(x)
          else:
             x = self.input2gate_conv(x)

          splitdim = 2
          x_a, x_b = x.split(x.size(splitdim) // 2, dim=splitdim)

          # Convert conditioning to gate
          c = self.conditioning2gate(x)
          c_a, c_b = c.split(c.size(splitdim) // 2, dim=splitdim)

          # Combine input and conditioning
          a, b = x_a + c_a , x_b + c_b
          x = torch.tanh(a) * torch.sigmoid(b) 

          # Convert combination to input again for residual connection
          x = self.combiner2input_conv(x)

          # Return
          return x


