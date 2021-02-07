import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

use_cuda = torch.cuda.is_available()


class AdvancedLSTMCell(nn.LSTMCell):
    # Extend LSTMCell to learn initial state
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class Caption(nn.Module):
    '''
    Mapping image_feat_grid X text_param ->att.grid
    (N,D_image,H,W) X (N,1,D_text) --> [N,1,H,W]
    '''
    def __init__(self, image_dim, text_dim, map_dim, vocab_size):
        super(Caption,self).__init__()
        self.map_size = map_dim
        self.vocab_size = vocab_size
        self.embedding_size = 196

        # creating image attention        
        self.conv1 = nn.Conv2d(image_dim,map_dim,kernel_size=1, dilation=2)

        # creating global represenation
        self.fc_global = nn.Linear(2048, 196)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size + self.map_size, self.embedding_size, 2, batch_first = True, bidirectional=True)
 
        self.linear = nn.Linear(self.map_size + 2*self.embedding_size, 512)
        self.dropout = nn.Dropout(p=0.4)
        self.word_projection = nn.Linear(512, self.vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.01
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.word_projection.bias.data.fill_(0)
        self.word_projection.weight.data.uniform_(-initrange, initrange)        

    def calculate_context(self, image_mapped, text_embedding):
        # image_feat N, 512, 14, 14
        # text embedding -> 196
        text_mapped = text_embedding.view(-1, 1, 14, 14).expand_as(image_mapped)
        elmtwize_mult = image_mapped * text_mapped

        atten_weights = torch.sum(elmtwize_mult, (2, 3))        
        context = F.normalize(atten_weights, p=2, dim=1)
        return context

    def forward(self, input_image_feat, sequences, sequences_lens):
        x = input_image_feat.permute(0, 3, 1, 2)
        image_mapped = self.conv1(x)  #(N, map_dim, H, W)
        image_global = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        image_global = self.fc_global(image_global)

        ## get intial context
        batch_size = input_image_feat.size(0)
        time_sequence = sequences.size(0)

        sequences = sequences.permute(1, 0)
        embedding = self.embedding(sequences)
 
        ## send global representation to the conext function
        ctx = self.calculate_context(image_mapped, image_global)

        logits = []
        generates = []
        hidden = None

        for i in range(time_sequence):
            out = torch.cat((embedding[:, i, :], ctx), 1) # Teacher forcing
         
            out = out.unsqueeze(1) 
            output, hidden = self.lstm(out, hidden)

            output = output.squeeze()
            word_output = self.linear(self.dropout(torch.cat((output, ctx),1)))
            word_output = F.log_softmax(self.word_projection(F.leaky_relu(word_output)))
           
            logits.append(word_output) 
            ctx = self.calculate_context(image_mapped, hidden[0][-1]) 
            
        logits = torch.stack(logits, 0)
        return logits  


'''
class DescribeModule(nn.Module):
    def __init__(self,output_num_choice, image_dim, text_dim, map_dim):
        super(DescribeModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.image_dim = image_dim
        self.text_fc = nn.Linear(text_dim, map_dim)
        self.att_fc_1 = nn.Linear(image_dim, map_dim)
        self.lc_out = nn.Linear(map_dim, self.out_num_choice)

    def forward_caption_only_image(self, input_image_feat, input_text):
        H, W = input_image_attention1.shape[2:4]
        att_softmax_1 = F.softmax(input_image_attention1.view(-1, H * W),dim=1).view(-1, 1, H*W)
        image_reshape = input_image_feat.view(-1,self.image_dim,H * W) #[N,image_dim,H*W]
        att_feat_1 = torch.sum(att_softmax_1 * image_reshape, dim=2)    #[N, image_dim]
        att_feat_1_mapped = self.att_fc_1(att_feat_1)       #[N, map_dim]

        text_mapped = self.text_fc(input_text)
        elmtwize_mult = att_feat_1_mapped * text_mapped  #[N, map_dim]
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1)
        scores = self.lc_out(elmtwize_mult)

        return scores
'''
