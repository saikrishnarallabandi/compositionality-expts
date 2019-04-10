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

class baseline_lstm(baseline_model):

        def __init__(self):
          super(baseline_lstm, self).__init__()

          self.encoder_fc = layers.SequenceWise(nn.Linear(80, 32))
          self.encoder_dropout = layers.SequenceWise(nn.Dropout(0.1))

          self.seq_model = nn.LSTM(32, 64, 1, bidirectional=True, batch_first=True)
          self.prefinal_fc = layers.SequenceWise(nn.Linear(128, 32))
          self.final_fc = nn.Linear(128, 64)

        def forward(self, c):
 
           c = c.transpose(1,2)
           #print("The type of local conditioning is ", c.type()) 
           x = self.encoder_fc(c)
           x = self.encoder_dropout(x)

           x, (h,c) = self.seq_model(x, None)
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)
           x = self.final_fc(hidden)

           return x

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


class VectorQuant(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=False):
        super().__init__()
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        print("Number of classes: ", n_classes)
        print("Length of vector: ", vec_len)

        ### Articulatory Features ###########
        self.embedding2arff = nn.Linear(vec_len,160)

        self.embedding_vc = nn.Parameter(torch.randn(n_channels, 3, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_vlng = nn.Parameter(torch.randn(n_channels, 5, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_vheight = nn.Parameter(torch.randn(n_channels, 5, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_vfront = nn.Parameter(torch.randn(n_channels, 5, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_vrnd = nn.Parameter(torch.randn(n_channels, 3, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_ctype = nn.Parameter(torch.randn(n_channels, 7, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_cpoa = nn.Parameter(torch.randn(n_channels, 8, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_cvox = nn.Parameter(torch.randn(n_channels, 3, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_asp = nn.Parameter(torch.randn(n_channels, 3, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_nuk = nn.Parameter(torch.randn(n_channels, 3, 16, requires_grad=True) * self.embedding_scale)
        self.embedding_arff = [ self.embedding_vc, self.embedding_vlng, self.embedding_vheight, self.embedding_vfront, self.embedding_vrnd, self.embedding_ctype, self.embedding_cpoa, self.embedding_cvox, self.embedding_asp, self.embedding_nuk]

        self.arff2embedding = nn.Linear(160, vec_len)
        ######################################

        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()

        # e: (160) x: (vec_len)
    def get_arff_embedding(self, x, e):

        x = self.embedding2arff(x)
        x = x.unsqueeze(1)
        #print("Shape of x: ", x.shape)
        #print("Shape of e: ", e.shape)
        #print("Shape of x - e: ", (x-e).shape) 
        #sys.exit()

        x_vc = x[:,:,:16]
        #print("Shapes of x_vc and embedding_vc: ", x_vc.shape, self.embedding_vc.shape, " and that of the argmin: ", (x_vc - self.embedding_vc).shape)
       

        x_vlng = x[:,:,16:32]
        #print("Shapes of x_vlng and embedding_vc: ", x_vlng.shape, self.embedding_vlng.shape, " and that of the argmin: ", (x_vlng - self.embedding_vlng).shape)       
     
        x_arff = x.split(16,dim=-1)
        c_arff = []
        for ctr, (a,b) in enumerate(list(zip(x_arff, self.embedding_arff))):
            #print("Shape of a and b: ", a.shape, b.shape)
            #print(ctr, (a-b).shape, (a-b).norm(dim=2).shape)
            c = (a-b).norm(dim=2).argmin(dim=1)
            #print("Shape of c: ", c.shape) #, c.view(c.shape[0]*c.shape[1])[0:10])
            #print("Shape of b: ", b.shape)  
            d = b.index_select(dim=1, index=c)
            #print("Shape of d: ", d.shape)
            c_arff.append(d)
          
        c_arff = torch.stack(c_arff, dim=-1)
        c_arff = c_arff.reshape(c_arff.shape[0]*c_arff.shape[1], c_arff.shape[2]*c_arff.shape[3])
        c_arff = self.arff2embedding(c_arff)
        #print("Shape of c_arff: ", c_arff.shape) 
        #print('\n')
        #sys.exit()
 
        return c_arff

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        #print("Shape of x input to the quantizer module: ", x.shape)
        x1 = x #.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        x1 = x.reshape(x.size(0)*x.size(1), x.size(2))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        #print("Shape of x after reshaping: ", x1.shape)

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        #for x1_chunk in x1.split(512, dim=0):
        for x1_chunk in x1.split(64, dim=0): 
            #print("Shape of supposed to be: ", (x1_chunk - embedding).norm(dim=3).argmin(dim=2).shape)
            arff_embedding = self.get_arff_embedding(x1_chunk, embedding)
            #index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
            index_chunks.append(arff_embedding)

        output_flat = torch.cat(index_chunks, dim=0)
        output = output_flat.view(x.size())
        #print("Shape of output: ", output.shape)

        entropy = 0

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=-1).pow(2)
        out2 = (x - output.detach()).float().norm(dim=-1).pow(2) + (x - x0).float().norm(dim=-1).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)
        sys.exit()

        # index: (N*samples, n_channels) long tensor
        #if True: # compute the entropy
        #    hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
        #    prob = hist.masked_select(hist > 0) / len(index)
        #    entropy = - (prob * prob.log()).sum().item()
        #    #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        #else:
        #    entropy = 0
        #index1 = (index + self.offset).view(index.size(0) * index.size(1))
        # index1: (N*samples*n_channels) long tensor
        #output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        #output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))
