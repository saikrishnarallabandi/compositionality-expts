import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys
import torch.nn.functional as F
import numpy as np

falcon_dir = '/home/sirisha.rallabandi/tools/falkon/'
sys.path.append(falcon_dir)
import src.nn.layers as layers


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
        self.update_z = layers.SequenceWise(nn.Linear(256, 128))
        self.update_c = layers.SequenceWise(nn.Linear(128+embed_size, 256))
        self.encoder_lstm = nn.LSTM(embed_size_image, embed_size_image)
        self.features2conditioning = nn.Linear(256,embed_size)
        self.latent2conditioning = nn.Linear(128,embed_size)
        self.combination2conditioning = nn.Linear(embed_size*2, embed_size)
        self.embedlatentcombination2embedding = nn.Linear(embed_size*2, embed_size)

    # Input shape( B, T, C) 
    # Generate T mus and T sigmas      
    def encoder(self, features):
        assert len(features.shape) == 3
        output = torch.tanh(self.encoder_fc(features))
        mu = self.image_linear_mu(output)
        sigma = self.image_linear_var(output)
        return mu, sigma

    def encoder_test(self, features, states=None):
        assert len(features.shape) == 3
        output = torch.tanh(self.encoder_fc(features))
        mu = self.image_linear_mu(output)
        sigma = self.image_linear_var(output)
        return mu, sigma, states


    def reparameterize(self, mu, sigma):

        std = torch.exp(0.5*sigma)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def sample_greedy_ConditionalZ_temporal(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []


        ## Send to the encoder and get features, Z (B, 1, C); (B, 1, C)
        features = self.image_linear(features)
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        conditioning = torch.tanh(self.features2conditioning(features))
        z = torch.tanh(self.latent2conditioning(z))

        # Prepare input to the decoder
        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3

        #_, states = self.lstm(conditioning)
        inputs = torch.cat((conditioning, z, embeddings), dim=1)

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


    def sample_greedy_ConditionalZ(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        
      
        ## Send to the encoder and get features, Z (B, 1, C); (B, 1, C)
        features = self.image_linear(features)
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        conditioning = self.features2conditioning(features)
        z = self.latent2conditioning(z)

        # Prepare input to the decoder
        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3

        _, states = self.lstm(conditioning)
        inputs = torch.cat((z, embeddings), dim=1)
        
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

    def forward_ConditionalZ_temporal(self, features, captions):
        
        # Do something about the image features (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)

        # Somehow generate a latent representation (B, 1, C)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        z = torch.tanh(self.latent2conditioning(z))

        # Somehow generate a conditional based on image features (B, 1, C)
        conditioning = torch.tanh(self.features2conditioning(features))

        # Embed the captions (B, T, C)
        embeddings = self.embed(captions)

        # Somehow combine the conditional and the latent representation (B, 2, C)
        inputs = torch.cat((conditioning, z, embeddings), dim = 1)        
        #print("Shape of inputs: ", inputs.shape)
 
        # Feed the combination of conditional, latent, captions through the decoder.  
        #_, states = self.lstm(conditioning)

        outputs, _ = self.lstm(inputs)
        outputs = self.final_linear(outputs)
 
        return outputs[:,0:-3,:], mu, sigma


    def forward_ConditionalZ_spatial(self, features, captions):

        # Do something about the image features (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)

        # Somehow generate a latent representation (B, 1, C)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
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
 
        return outputs[:,:-1,:], mu, sigma


    def sample_greedy_ConditionalZ_spatial(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
       
        ## Send to the encoder and get features, Z (B, 1, C); (B, 1, C)
        features = self.image_linear(features)
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        conditioning = F.relu(self.features2conditioning(features))
        z = F.relu(self.latent2conditioning(z))

        # Prepare input to the decoder
        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3

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


    def forward_ConditionalZ_spatiotemporal(self, features, captions):

        # Do something about the image features (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)

        # Somehow generate a latent representation (B, 1, C)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        z = self.latent2conditioning(z)

        # Somehow generate a conditional based on image features (B, 1, C)
        conditioning = self.features2conditioning(features)

        # Embed the captions (B, T, C)
        embeddings = self.embed(captions)

        # Upsample the latent representation
        assert len(z.shape) == 3
        z = z.transpose(1,2)
        z = F.interpolate(z, size=[z.shape[-1]*captions.shape[1]])
        z = z.transpose(1,2)

        # Somehow combine the embeddings and the latent representation (B, T, C)
        inputs = torch.cat((z, embeddings), dim = -1)        
        inputs = torch.tanh(self.embedlatentcombination2embedding(inputs))

        # Feed the combination of conditional, latent, captions through the decoder.  
        _, states = self.lstm(conditioning)
        outputs, _ = self.lstm(inputs, states)
        outputs = self.final_linear(outputs)
 
        return outputs[:,:-1,:], mu, sigma


    def sample_greedy_ConditionalZ_spatiotemporal(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        
      
        ## Send to the encoder and get features, Z (B, 1, C); (B, 1, C)
        features = self.image_linear(features)
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        conditioning = self.features2conditioning(features)
        z = self.latent2conditioning(z)

        # Prepare input to the decoder
        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3

        _, states = self.lstm(conditioning)
        inputs = torch.cat((embeddings, z), dim = -1)
        inputs = torch.tanh(self.embedlatentcombination2embedding(inputs))

        for i in range(self.max_seg_length):
            #print("Shape of inputs: ", inputs.shape)
            assert inputs.shape[1] == 1
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
            inputs = torch.cat((inputs, z), dim=-1)
            inputs = torch.tanh(self.embedlatentcombination2embedding(inputs))
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

