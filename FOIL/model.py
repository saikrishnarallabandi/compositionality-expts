import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class RNN_model(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, out_classes, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(RNN_model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_classes)

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        features = self.bn(self.image_linear(features))

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        hiddens, _ = pad_packed_sequence(hiddens,lengths)
        outputs = self.linear(hiddens[:, 0, :])
        return outputs
   
 
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
        self.image_linear_mu = nn.Linear(embed_size_image, embed_size_image)
        self.image_linear_var = nn.Linear(embed_size_image, embed_size_image)

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
        features = self.bn(self.image_linear(features))

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
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

