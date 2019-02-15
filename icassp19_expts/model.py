import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys
import torch.nn.functional as F
import numpy as np

falcon_dir = '/home/srallaba/projects/caption_generation/repos/falkon'
sys.path.append(falcon_dir)
import src.nn.layers as layers


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
        features = self.image_linear(features)
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            #print(predicted.shape)
            if predicted.item() == 2:
               break
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids



class CaptionRNN_nopadpack(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN_nopadpack, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = layers.SequenceWise(nn.Linear(hidden_size, vocab_size))
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        features = self.bn(self.image_linear(features))

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        outputs, _ = self.lstm(embeddings)
        outputs = self.linear(outputs)
        return outputs[:,:-1,:]
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = self.image_linear(features)
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            assert len(inputs.shape) == 3
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens)            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(-1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            #print(predicted.shape)
            if predicted.item() == 2:
               break
            inputs = self.embed(predicted.squeeze(0))                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class CaptionRNN_VI(nn.Module):
    def __init__(self, feature_size, embed_size_image, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN_VI, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size_image)
        self.bn = nn.BatchNorm1d(embed_size_image, momentum=0.01)
        self.image_linear_mu = SequenceWise(nn.Linear(embed_size_image, embed_size_image))
        self.image_linear_var = SequenceWise(nn.Linear(embed_size_image, embed_size_image))

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        features = self.bn(self.image_linear(features))

        mu = self.image_linear_mu(features)
        sigma = self.image_linear_var(features)

        std = torch.exp(0.5*sigma)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        z = torch.cat((z, features), 1)

        embeddings = self.embed(captions)
        embeddings = torch.cat((z.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs, mu, sigma

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = self.image_linear(features)

        mu = self.image_linear_mu(features)
        sigma = self.image_linear_var(features)

        std = torch.exp(0.5*sigma)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        z = torch.cat((z, features), 1)

        inputs = z.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            if predicted.item() == 2:
               break
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids



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
        
        self.image_linear = nn.Linear(feature_size, embed_size_image)
        self.bn = nn.BatchNorm1d(embed_size_image, momentum=0.01)
        self.image_linear_mu = layers.SequenceWise(nn.Linear(embed_size_image, 128))
        self.image_linear_var = layers.SequenceWise(nn.Linear(embed_size_image, 128))
        self.update_z = layers.SequenceWise(nn.Linear(256, 128))
        self.update_c = layers.SequenceWise(nn.Linear(128+embed_size, 256))
        self.encoder_lstm = nn.LSTM(embed_size_image, embed_size_image)
        self.features2conditioning = nn.Linear(256,embed_size)
        self.latent2conditioning = nn.Linear(128,embed_size)

    # Input shape( B, T, C) 
    # Generate T mus and T sigmas      
    def encoder(self, features):
        assert len(features.shape) == 3
        output, _ = self.encoder_lstm(features, None)
        mu = self.image_linear_mu(output)
        sigma = self.image_linear_var(output)
        return mu, sigma

    def encoder_test(self, features, states=None):
        assert len(features.shape) == 3
        output, states = self.encoder_lstm(features, states)
        mu = self.image_linear_mu(output)
        sigma = self.image_linear_var(output)
        return mu, sigma, states


    def reparameterize(self, mu, sigma):

        std = torch.exp(0.5*sigma)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward_temporalz(self, features, captions):

        max_length = captions.shape[-1]
        #print("Max length is ", max_length)
        mus = []
        sigmas = []
        # Repeat the features max_length times and send to encoder
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        mus.append(mu)
        sigmas.append(sigma)
        inputs = captions[:,0]
        inputs = inputs.unsqueeze(1)
        #print("Shape of inputs: ", inputs.shape)
        embeddings = self.embed(inputs)
        assert len(embeddings.shape) == 3
        states = None
        inputs = torch.cat((z, embeddings), -1)

        logits = []
        for i in range(max_length):
            assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)
            outputs = self.final_linear(outputs) 
            logits.append(outputs.squeeze(1)) 
            _, predicted = outputs.max(-1)
            inputs = self.embed(predicted.squeeze(0))  
            mu, sigma, states_encoder = self.encoder_test(features, states_encoder)
            z = self.reparameterize(mu, sigma)
            mus.append(mu)
            sigmas.append(sigma)
            assert len(inputs.shape) == 3
            inputs = torch.cat((z, inputs), -1)
        #print("Shape of logits, mu, sigma ", logits[0].shape, mu[1].shape, sigma[2].shape)
        logits = torch.stack(logits, 1)
        mus = torch.stack(mus, 1)
        sigmas = torch.stack(sigmas, 1)
        return logits[:,1:,:], mus, sigmas   

    def forward_TFspatialz(self, features, captions):
        """Decode image feature vectors and generates captions."""

        max_length = captions.shape[-1] - 1
        mus = []
        sigmas = []

        ## Send to the encoder and get Z0 (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        mus.append(mu)
        sigmas.append(sigma)
        #print("Shape of z: ", z.shape)

       	## Embed the captions (	B, T, C )
        embeddings = self.embed(captions)
        inputs = captions[:,0]
        inputs = inputs.unsqueeze(1)
        embeddings = self.embed(inputs)
        assert len(embeddings.shape) == 3
        states = None
        inputs = torch.cat((z, embeddings), -1)
        #print("Shape of inputs: ", inputs.shape)

        logits = []
        for i in range(max_length):
            assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)
            outputs = self.final_linear(outputs)
            logits.append(outputs.squeeze(1))
            inputs = self.embed(captions[:,i+1].unsqueeze(1))
            assert len(inputs.shape) == 3
            inputs = torch.cat((z, inputs), -1)
        logits = torch.stack(logits, 1)
        mus = torch.stack(mus, 1)
        sigmas = torch.stack(sigmas, 1)
        return logits, mus, sigmas

    def forward_TFtemporalz(self, features, captions):
        """Decode image feature vectors and generates captions."""

        max_length = captions.shape[-1] - 1
        mus = []
        sigmas = []

        ## Send to the encoder and get Z0 (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        mus.append(mu)
        sigmas.append(sigma)
        #print("Shape of z and features: ", z.shape, features.shape)

        ## Embed the captions ( B, T, C )
        embeddings = self.embed(captions)
        inputs = captions[:,0]
        inputs = inputs.unsqueeze(1)
        embeddings = self.embed(inputs)
        assert len(embeddings.shape) == 3
        states = None
        #features = self.features2conditioning(features)
        #inputs = torch.cat((features, embeddings), -1)
        inputs = features
        #print("Shape of inputs, embeddings: ", inputs.shape, embeddings.shape)

        logits = []
        for i in range(max_length):
            assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)
            outputs = self.final_linear(outputs)
            logits.append(outputs.squeeze(1))
            inputs = self.embed(captions[:,i+1].unsqueeze(1))
            assert len(inputs.shape) == 3
            #print("Shape of inputs and z: ", inputs.shape, z.shape)
            # Generate new z
            z =  torch.tanh(self.update_c(torch.cat((z, inputs), -1)))
            #inputs = torch.cat((z, inputs), -1)
            inputs = z
            z = torch.tanh(self.update_z(z))
            
        logits = torch.stack(logits, 1)
        mus = torch.stack(mus, 1)
        sigmas = torch.stack(sigmas, 1)
        return logits, mus, sigmas


    def forward_ConditionalZ(self, features, captions):
        """Decode image feature vectors and generates captions."""

        max_length = captions.shape[-1] - 1

        ## Send to the encoder and get features, Z (B, 1, C); (B, 1, C)
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        conditioning = self.features2conditioning(features)
        z = self.latent2conditioning(z)
        #print("Shape of z and features: ", z.shape, features.shape)

        ## Embed the captions ( B, T, C )
        embeddings = self.embed(captions)

        # Prepare input to the decoder
        #print("Shape of features, z, embeddings: ", features.shape, z.shape, embeddings.shape)
        inputs = torch.cat((conditioning, z), dim = 1)
        outputs, states = self.lstm(inputs)
        inputs = outputs[:, -1, :].unsqueeze(1)
        inputs = torch.cat((inputs, embeddings), dim = 1)        
        #print("Shape of inputs: ", inputs.shape)

        assert len(embeddings.shape) == 3
        outputs, _ = self.lstm(inputs)
        outputs = self.final_linear(outputs)
        #print("Shape of outputs: ", outputs.shape)
        return outputs[:,:-2,:], mu, sigma



    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""

        max_length = captions.shape[-1]
        #print("Max length is ", max_length)

        # Repeat the features max_length times and send to encoder
        features = self.bn(self.image_linear(features))
        features = features.unsqueeze(1)
        features = features.transpose(1,2)
        features = F.interpolate(features, size=[max_length])
        features_repeated = features.transpose(1,2)

        mu, sigma = self.encoder(features_repeated)
        #print("Shape of features from the encoder: ", mu.shape, sigma.shape)
        z = self.reparameterize(mu, sigma)
        #print("Shape of z: ", z.shape)

        embeddings = self.embed(captions)
        #print("Shape of embeddings and z: ", embeddings.shape, z.shape)

        embeddings = torch.cat((z, embeddings), -1)
        #print("Shape of concatenated embedding: ", embeddings.shape)

        outputs, _ = self.lstm(embeddings)
        outputs = self.final_linear(outputs)
        #print("Shape of outputs: ", outputs.shape)

        return outputs[:,:-1,], mu, sigma

    def sample_choice(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = self.image_linear(features)

        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        assert len(z.shape) == 3
        
        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3
        states = None
        inputs = torch.cat((z, embeddings), -1)
        for i in range(self.max_seg_length):
            assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.final_linear(outputs).squeeze(0)            # outputs:  (batch_size, vocab_size)
            #print("Shape of outputs: ", outputs.shape)
            logits = F.softmax(outputs.squeeze(0), 0)
            choice = np.random.choice(np.arange(logits.shape[0]), p = logits.cpu().numpy())
            predicted = torch.LongTensor([choice]).unsqueeze(0).cuda()
            #_, predicted = outputs.max(-1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            #print("Shape of predicted is : ", predicted, predicted.shape)
            if predicted.item() == 2:
               break
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            #inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            mu, sigma, states_encoder = self.encoder_test(features, states_encoder)
            z = self.reparameterize(mu, sigma)
            inputs = torch.cat((z, inputs), -1)
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

        inputs = torch.cat((conditioning, z), dim = 1)
        states = None
        outputs, states = self.lstm(inputs, states)
        inputs = outputs[:, -1, :].unsqueeze(0)
        
        for i in range(self.max_seg_length):
            assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
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

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = self.image_linear(features)

        features = features.unsqueeze(1)
        mu, sigma, states_encoder = self.encoder_test(features)
        z = self.reparameterize(mu, sigma)
        assert len(z.shape) == 3

        inputs = torch.rand(z.shape[0], 1)
        inputs.zero_()
        inputs[:,0] = 1
        embeddings = self.embed(inputs.long().cuda())
        #print("Shape of embeddings: ", embeddings.shape)
        assert len(embeddings.shape) == 3
        states = None
        #features = self.features2conditioning(features)
        #inputs = torch.cat((features, embeddings), -1)
        inputs = features
        for i in range(self.max_seg_length):
            assert inputs.shape[1] == 1
            assert len(inputs.shape) == 3
            outputs, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.final_linear(outputs)            # outputs:  (batch_size, vocab_size)
            #print("Shape of outputs: ", outputs.shape)
            _, predicted = outputs.max(-1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            #print("Shape of predicted is : ", predicted, predicted.shape)
            if predicted.item() == 2:
               break
            inputs = self.embed(predicted.squeeze(0))                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            #mu, sigma, states_encoder = self.encoder_test(features, states_encoder)
            #z = self.reparameterize(mu, sigma)

            # Generate new z
            z =  torch.tanh(self.update_c(torch.cat((z, inputs), -1)))
            #inputs = torch.cat((z, inputs), -1)
            inputs = z
            z = torch.tanh(self.update_z(z))

            #z =  torch.tanh(self.update_z(torch.cat((z, inputs), -1)))
            #inputs = torch.cat((z, inputs), -1)

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

