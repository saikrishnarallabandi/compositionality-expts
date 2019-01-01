import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys
from layers import *
from modules import *
import numpy as np

class CaptionRNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        features = self.bn(self.image_linear(features))

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class CaptionCNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        layers = 12
        stacks = 4
        layers_per_stack = layers // stacks
        self.kernel_size = 3
        self.stride = 1
        self.vocab_size = vocab_size
           

        self.conv_modules = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            self.padding = int((self.kernel_size - 1) * dilation)
            conv = residualconvmodule(embed_size,embed_size, self.kernel_size, self.stride, self.padding,dilation)
            self.conv_modules.append(conv)


        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        self.final_fc1 = SequenceWise(nn.Linear(256, 512))
        self.final_fc2 = SequenceWise(nn.Linear(512, vocab_size))


        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        features = self.bn(self.image_linear(features))

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        c = None
        x = embeddings.transpose(1,2)
        for module in self.conv_modules:
          x = F.relu(module(x, c))
        x = x.transpose(1,2)

        x = F.relu(self.final_fc1(x))
        x = self.final_fc2(x)

        return x[:,:-1,:]
    
    def clear_buffers(self):

       for module in self.conv_modules:
           module.clear_buffer()

    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""

        outputs = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        bsz = inputs.shape[0]
        
        self.clear_buffers()
        c = None
        x = inputs
        for i in range(self.max_seg_length):
            module_count = 0
            for module in self.conv_modules:
                module_count += 1
                #print("  Module: Feeding into the module number: ", module_count)
                x = F.relu(module.incremental_forward(x, c))
            x = F.relu(self.final_fc1(x))
            x = self.final_fc2(x)
            #print("Shape of x", x.shape)

            v, predicted = torch.max(x,2)
            outputs.append(predicted[0])
            x = self.embed(predicted)
            #print(predicted, predicted.shape, v)
            
        outputs = torch.stack(outputs, 1)
        return outputs

class CaptionSingleCNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=60):
        """Set the hyper-parameters and build the layers."""
        super(CaptionSingleCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.kernel_size = 3
        self.stride = 1
        self.vocab_size = vocab_size
        self.padding = 1
        self.conv = residualconvmodule(embed_size, embed_size, self.kernel_size, self.stride, self.padding, dilation = 1)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.final_fc1 = SequenceWise(nn.Linear(256, 512))
        self.final_fc2 = SequenceWise(nn.Linear(512, vocab_size))

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        features = self.bn(self.image_linear(features))

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        c = None
        x = embeddings.transpose(1,2)
        x = self.conv(x,c)
        x = x.transpose(1,2)

        x = F.relu(self.final_fc1(x))
        x = self.final_fc2(x)

        return x[:,:-1,:]

    def clear_buffers(self):
        self.conv.clear_buffer()



    def sample(self, features, states=None, return_logits=0):
        """Generate captions for given image features using greedy search."""

        outputs = []
        logits = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        bsz = inputs.shape[0]

        self.clear_buffers()
        c = None
        x = inputs
        for i in range(self.max_seg_length):
            x = F.relu(self.conv.incremental_forward(x, c))
            x = F.relu(self.final_fc1(x))
            x = self.final_fc2(x)
            logits.append(x.squeeze(1))
            v, predicted = torch.max(x,2)
            outputs.append(predicted[0])
            x = self.embed(predicted)

        outputs = torch.stack(outputs, 1)
        logits = torch.stack(logits)
        if return_logits:
            #print("Returning logits")
            return logits
        return outputs


