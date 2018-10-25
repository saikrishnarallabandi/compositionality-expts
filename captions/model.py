
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class VED(nn.Module):

    def __init__(self,encoder, decoder):
       super(VED, self).__init__()
       self.encoder = EncoderCNN()
       self.decoder = DecoderRNN()
       self.init_hidden()
       
    def forward(self,images):
       mu, log_var = self.encoder(images)
       z = self.reparameterize(mu, log_var) 
       decoded = self.decoder(z)
       return decoded,mu,log_var

    def reparameterize(self, mu, log_var):
       std = torch.exp(0.5*log_var)
       eps = torch.rand_like(std)
       return eps.mul(std).add_(mu)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        temp1 = torch.nn.Parameter(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
        temp2 = torch.nn.Parameter(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
        temp = (temp1, temp2)
        logging.debug("I have initialized hidden as {}".format(temp1.shape))
        if self.rnn_type == 'LSTM':
            return temp


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
         captions = captions[:,:-1]
         embeddings = self.embed(captions)
         inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
         hiddens, _ = self.lstm(inputs)
         outputs = self.linear(hiddens)
         return outputs



    def sample(self, inputs, states=None, max_len=20):
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids


    def sample_beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            all_candidates = []
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]



 



