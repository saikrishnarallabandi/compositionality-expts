import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

print_flag = 0

class VED(nn.Module):

    def __init__(self,encoder, decoder):
       super(VED, self).__init__()
       self.encoder = encoder #EncoderCNN()
       self.decoder = decoder #DecoderRNN()
       #self.init_hidden()
       
    def forward(self, features, captions=None):
       mu, log_var = self.encoder.forward_nofeatextraction(features)
       z = self.reparameterize(mu, log_var)

       #if captions is None:
       #    return mu, log_var, z
 
       if print_flag:
          print("Shape of z before going into decoder is ", z.shape )
       decoded = self.decoder(z, captions)
       return decoded,mu,log_var,z

    def reparameterize(self, mu, log_var):
       std = torch.exp(0.5*log_var)
       eps = torch.rand_like(std)
       return eps.mul(std).add_(mu)

    def init_hidden(self):
        weight = next(self.parameters())
        temp1 = torch.nn.Parameter(torch.zeros(self.nlayers, self.bsz, self.nhid).cuda())
        temp2 = torch.nn.Parameter(torch.zeros(self.nlayers, self.bsz, self.nhid).cuda())
        temp = (temp1, temp2)
        logging.debug("I have initialized hidden as {}".format(temp1.shape))
        if self.rnn_type == 'LSTM':
            return temp

    def sample(self, features):
        mu, log_var = self.encoder.forward_nofeatextraction_sample(features)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder.sample(z) 
        return decoded

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, encoder_output_dim):
        super(EncoderCNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoder_output_dim = encoder_output_dim

        #self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.embed = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.bottle_neck = nn.Linear(embed_size, 512)
        
        self.fc1 = nn.Linear(512, encoder_output_dim) #nn.Linear(batch_size, embed_size)
        self.fc2 = nn.Linear(512, encoder_output_dim) #nn.Linear(batch_size, embed_size)

    def print_shapes(self):
        print("Shape of embedding in encoder: ", self.embed_size)
        print("Shape of hidden size in encoder: ", self.hidden_size)
        print("Shape of encoder output:  ", self.encoder_output_dim)
  
    def forward(self, images):

        if print_flag:
           self.print_shapes()
        features = self.embed(features)
        features = self.bn(features)

        features = self.bottle_neck(features) 

        mu = self.fc1(features)
        log_var = self.fc2(features)

        return mu, log_var

    def forward_nofeatextraction_sample(self, features):

        features = self.embed(features)
        #features = self.bn(features)

        features = self.bottle_neck(features) 

        mu = self.fc1(features)
        log_var = self.fc2(features)

        return mu, log_var

    def forward_nofeatextraction(self, features):

        if print_flag:
           self.print_shapes()
           print("Shape of features before embedding them in encoder ", features.shape)
        features.squeeze_(0)
        features = self.embed(features)
        features = self.bn(features)
        if print_flag:
           print("Shape of features after embedding them in encoder ", features.shape)

        features = self.bottle_neck(features) 

        mu = self.fc1(features)
        log_var = self.fc2(features)

        if print_flag:
           print("Shape of mu and log_var: ", mu.shape, log_var.shape)

        return mu, log_var


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_output_dim, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size+encoder_output_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.encoder_output_dim = encoder_output_dim

    def forward(self, features, captions):
         if print_flag:
            print("Shape of features and captions in decoder: ", features.shape, captions.shape)
         captions.squeeze_(0)
         #captions = captions[:,:-1]
         embeddings = self.embed(captions)
         if print_flag:
            print("Shape of embeddings and features : ", embeddings.shape, features.shape)

         # Unsqueeze the features
         features.unsqueeze_(1)
         # Repeat the features
         length_desired = captions.shape[1]
         features = features.repeat(1, length_desired, 1)
         if print_flag:
            print("Shape of embeddings and features right before cating them: ", embeddings.shape, features.shape)


         inputs = torch.cat((features, embeddings), -1)
         hiddens, _ = self.lstm(inputs)
         outputs = self.linear(hiddens)
         if print_flag:
            print("Shape of the outputs I am returning: ", outputs.shape)
         return outputs

    def sample(self, features, states=None, max_len=20):
         captions = features.new(features.shape[0],1)
         captions.zero_()
         if print_flag:
            print("Shape of features and captions in decoder: ", features.shape, captions.shape)

         embeddings = self.embed(captions.long())
         captions.unsqueeze_(1)
         features.unsqueeze_(1)  
         if print_flag:
            print("Shape of embeddings and features : ", embeddings.shape, features.shape)
         inputs = torch.cat((features, embeddings), -1)

         sampled_ids = []
         for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            if print_flag:
               print("Shape of outputs and predicted: ", outputs.shape, predicted.shape)
            sampled_ids.append(predicted.item())
            embeddings = self.embed(predicted).unsqueeze_(0)
            if print_flag:
               print("Shape of features and embeddings: ", features.shape, embeddings.shape)
            inputs = torch.cat((features, embeddings), -1)
         return sampled_ids
