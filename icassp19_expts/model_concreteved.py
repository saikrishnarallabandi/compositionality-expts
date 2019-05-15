import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys
import torch.nn.functional as F
import numpy as np

falcon_dir = '/home/sirisha.rallabandi/tools/falkon/'
sys.path.append(falcon_dir)
import src.nn.layers as layers

EPS=1e-12

class CaptionRNN_VItopline(nn.Module):
    def __init__(self, feature_size, embed_size_image, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN_VItopline, self).__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(128+embed_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.final_linear = layers.SequenceWise(nn.Linear(hidden_size, vocab_size))
        self.max_seg_length = max_seq_length
        self.encoder_fc = nn.Linear(embed_size_image, embed_size_image) 
        self.image_linear = nn.Linear(feature_size, embed_size_image)
        self.bn = nn.BatchNorm1d(embed_size_image, momentum=0.01)
        self.image_linear_mu = layers.SequenceWise(nn.Linear(embed_size_image, 128))
        self.image_linear_var = layers.SequenceWise(nn.Linear(embed_size_image, 128))
        self.image_linear_alpha = layers.SequenceWise(nn.Linear(embed_size_image, 128))
        self.discretez_fc = layers.SequenceWise(nn.Linear(128, 128))
        self.update_z = layers.SequenceWise(nn.Linear(256, 128))
        self.update_c = layers.SequenceWise(nn.Linear(128+embed_size, 256))
        self.encoder_lstm = nn.LSTM(embed_size_image, embed_size_image)
        self.features2conditioning = nn.Linear(256,embed_size)
        self.latent2conditioning = nn.Linear(256,embed_size)
        self.combination2conditioning = nn.Linear(embed_size*2, embed_size)
        self.embedlatentcombination2embedding = nn.Linear(embed_size*2, embed_size)
        self.temperature = 0.67

    # Input shape( B, T, C) 
    # Generate T mus and T sigmas      
    def encoder(self, features):
        assert len(features.shape) == 3
        output = torch.tanh(self.encoder_fc(features))
        mu = self.image_linear_mu(output)
        sigma = self.image_linear_var(output)
        alpha = self.image_linear_alpha(output)
        return mu, sigma, alpha

    def reparameterize(self, mu, sigma):

        if self.training:
           std = torch.exp(0.5*sigma)
           eps = torch.rand_like(std)
           z = eps.mul(std).add_(mu)
           return z
        else:
          return mu

    def reparameterize_discrete(self, encoded):
        alpha = self.discretez_fc(encoded)
        #print("Shape of alpha: ", alpha.shape)
        alpha = F.softmax(alpha, dim=0)
        #print("Shape of alpha: ", alpha.shape)
        #return alpha

        if self.training:
            return alpha
            unif = torch.rand(alpha.size())
            if torch.cuda.is_available():
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
         
        else:
            alpha = alpha.squeeze(1)  
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            a = torch.ones(1)
            one_hot_samples[max_alpha] =  1 #a.long()
            if torch.cuda.is_available():
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples.unsqueeze(1)    

    def forward_ConditionalZ_spatial(self, features, captions):

        # Do something about the image features (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)

        # Somehow generate a latent representation (B, 1, C)
        mu, sigma, alpha = self.encoder(features)
        z = self.reparameterize(mu, sigma)
        z_discrete = self.reparameterize_discrete(alpha)
        assert len(z.shape) + len(z_discrete.shape) == 6
        z = torch.cat((z, z_discrete), dim=-1)   
        z = F.relu(self.latent2conditioning(z))
          

        # Somehow generate a conditional based on image features (B, 1, C)
        conditioning = F.relu(self.features2conditioning(features))

        # Embed the captions (B, T, C)
        embeddings = self.embed(captions)

        # Somehow combine the conditional and the latent representation (B, 1, C)
        combination = torch.cat((conditioning, z), dim = -1)        
        conditioning = F.relu(self.combination2conditioning(combination))

        # Feed the combination of conditional, latent, captions through the decoder.  
        _, states = self.lstm(conditioning)

        outputs, _ = self.lstm(embeddings, states)
        outputs = self.final_linear(outputs)
 
        return outputs[:,:-1,:], mu, sigma, z_discrete


    def sample_greedy_ConditionalZ_spatial(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
       
        ## Send to the encoder and get features, Z (B, 1, C); (B, 1, C)
        features = self.image_linear(features)
        features = features.unsqueeze(1)
        mu, sigma, alpha = self.encoder(features)
        z = self.reparameterize(mu, sigma)
       	z_discrete = self.reparameterize_discrete(alpha)
       	assert len(z.shape) + len(z_discrete.shape) == 6
       	z = torch.cat((z, z_discrete), dim=-1)
        z = F.relu(self.latent2conditioning(z))
 
        # Prepare input to the decoder
        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3

        # Somehow generate a conditional based on image features (B, 1, C)
        conditioning = F.relu(self.features2conditioning(features))
        combination = torch.cat((conditioning, z), dim = -1)        
        conditioning = F.relu(self.combination2conditioning(combination))

        _, states = self.lstm(conditioning)
        inputs = embeddings
        
        for i in range(self.max_seg_length):
            print("Shape of inputs: ", inputs.shape)
            #assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            if i == 0:
               outputs = outputs[:,-1,:].unsqueeze(0)
            outputs = self.final_linear(outputs)            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(-1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            #print("Shape of predicted is : ", predicted, predicted.shape)
            if predicted.item() == 2:
               break
            inputs = self.embed(predicted.squeeze(0))                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


