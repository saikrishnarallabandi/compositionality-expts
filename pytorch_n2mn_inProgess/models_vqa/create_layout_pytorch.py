import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory

'''
H_feat = 14
W_feat = 14
D_feat = 2048
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 1000
num_layers = 2
encoder_dropout = True
decoder_dropout = True
decoder_sampling = False
T_encoder = 26
T_decoder = 13
N = 64
use_qpn = True
qpn_dropout = True
reduce_visfeat_dim = False
glove_mat_file = './exp_vqa/data/vocabulary_vqa_glove.npy'


num_vocab_txt:  17742  embed_dim_txt=  300  num_vocab_nmn=  5 decoder_sampling=  False  num_choices=  3001
'''

class AdvancedLSTM(nn.LSTM):
    # Class for learning initial hidden states when using LSTMs
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, input, hx=None):
        if hx is None:
            n = input.batch_sizes[0]
            hx = self.initial_state(n)
        return super(AdvancedLSTM, self).forward(input, hx=hx)


class pLSTM(AdvancedLSTM):
    # Pyramidal LSTM
    def __init__(self, *args, **kwargs):
        super(pLSTM, self).__init__(*args, **kwargs)
        self.shuffle = SequenceShuffle()

    def forward(self, input, hx=None):
        return super(pLSTM, self).forward(self.shuffle(input), hx=hx)


class EncoderModel(nn.Module):
    # Encodes utterances to produce keys and values
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.rnns = nn.ModuleList()
        encoder_dim = 300
        questions_vocab_size = 17742
        lstm_dim = 512
        value_dim = 128
        key_dim = 128

        self.token_embeddings = torch.nn.Embedding(questions_vocab_size, encoder_dim)
        self.rnns.append(AdvancedLSTM(encoder_dim, lstm_dim, bidirectional=True))
        #self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        #self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        #self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        self.key_projection = nn.Linear(lstm_dim * 2, key_dim)
        self.value_projection = nn.Linear(lstm_dim * 2, value_dim)

    def forward(self, utterances, utterance_lengths):
        h = utterances

        # Printing the values

        # Pass through the embedding layer
        h = self.token_embeddings(h)

        # Sort and pack the inputs
        sorted_lengths, order = torch.sort(utterance_lengths, 0, descending=True)
        _, backorder = torch.sort(order, 0)
        h = h[:, order, :]
        h = pack_padded_sequence(h, sorted_lengths.data.cpu().numpy())

        # RNNs
        for rnn in self.rnns:
            h, _ = rnn(h)

        # Unpack and unsort the sequences
        h, output_lengths = pad_packed_sequence(h)
        h = h[:, backorder, :]
        output_lengths = torch.from_numpy(np.array(output_lengths))
        if backorder.data.is_cuda:
            output_lengths = output_lengths.cuda()
        output_lengths = output_lengths[backorder.data]

        # Apply key and value
        keys = self.key_projection(h)
        values = self.value_projection(h)

        return keys, values, output_lengths


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_argmax(logits, dim):
    # Draw from a multinomial distribution efficiently
    return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


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


def output_mask(maxlen, lengths):
    """
    Create a mask on-the-fly
    :param maxlen: length of mask
    :param lengths: length of each sequence
    :return: mask shaped (maxlen, len(lengths))
    """
    lens = lengths.unsqueeze(0)
    ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
    mask = ran < lens
    return mask


def calculate_attention(keys, mask, queries):
    """
    Attention calculation
    :param keys: (N, L, key_dim)
    :param mask: (N, L)
    :param queries: (N, key_dim)
    :return: attention (N, L)
    """
    energy = torch.bmm(keys, queries.unsqueeze(2)).squeeze(2) * mask  # (N, L)
    energy = energy - (1 - mask) * 1e4  # subtract large number from padded region
    emax = torch.max(energy, 1)[0].unsqueeze(1)  # (N, L)
    eval = torch.exp(energy - emax) * mask  # (N, L)
    attn = eval / (eval.sum(1).unsqueeze(1))  # (N, L)
    return attn


def calculate_context(attn, values):
    """
    Context calculation
    :param attn:  (N, L)
    :param values: (N, L, value_dim)
    :return: Context (N, value_dim)
    """
    ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)  # (N, value_dim)
    return ctx


class DecoderModel(nn.Module):
    # Speller/Decoder
    def __init__(self):
        super(DecoderModel, self).__init__()
        tokens_vocab_size = 1+5 # start token
        decoder_dim = 300 
        value_dim = 128
        key_dim = 128
        teacher_force_rate = 1.0

        self.embedding = nn.Embedding(tokens_vocab_size, decoder_dim)
        self.input_rnns = nn.ModuleList()
        self.input_rnns.append(AdvancedLSTMCell(decoder_dim + value_dim, decoder_dim))
        self.input_rnns.append(AdvancedLSTMCell(decoder_dim, decoder_dim))
        self.input_rnns.append(AdvancedLSTMCell(decoder_dim, decoder_dim))
        self.query_projection = nn.Linear(decoder_dim, key_dim)
        self.char_projection = nn.Sequential(
            nn.Linear(decoder_dim+value_dim, decoder_dim),
            nn.LeakyReLU(),
            nn.Linear(decoder_dim, tokens_vocab_size)
        )
        self.force_rate = teacher_force_rate
        self.char_projection[-1].weight = self.embedding.weight  # weight tying

    def forward_pass(self, input_t, keys, values, mask, ctx, input_states):
        # Embed the previous character
        embed = self.embedding(input_t)
        # Concatenate embedding and previous context
        ht = torch.cat((embed, ctx), dim=1)
        # Run first set of RNNs
        new_input_states = []
        for rnn, state in zip(self.input_rnns, input_states):
            ht, newstate = rnn(ht, state)
            new_input_states.append((ht, newstate))
        # Calculate query
        query = self.query_projection(ht)
        # Calculate attention
        attn = calculate_attention(keys=keys, mask=mask, queries=query)
        # Calculate context
        ctx = calculate_context(attn=attn, values=values)
        # Concatenate hidden state and context
        ht = torch.cat((ht, ctx), dim=1)
        # Run projection
        logit = self.char_projection(ht)
        # Sample from logits
        generated = gumbel_argmax(logit, 1)  # (N,)
        return logit, generated, ctx, attn, new_input_states

    def forward(self, inputs, input_lengths, keys, values, utterance_lengths, future=0):
        mask = Variable(output_mask(values.size(0), utterance_lengths).transpose(0, 1)).float()
        if torch.cuda.is_available():
            mask = mask.cuda()
        values = values.transpose(0, 1)
        keys = keys.transpose(0, 1)
        t = inputs.size(0)
        n = inputs.size(1)

        # Initial states
        input_states = [rnn.initial_state(n) for rnn in self.input_rnns]

        # Initial context
        h0 = input_states[-1][0]
        query = self.query_projection(h0)
        attn = calculate_attention(keys, mask, query)
        ctx = calculate_context(attn, values)

        # Decoder loop
        logits = []
        attns = []
        generateds = []
        for i in range(t):
            # Use forced or generated inputs
            if len(generateds) > 0 and self.force_rate < 1 and self.training:
                input_forced = inputs[i]
                input_gen = generateds[-1]
                input_mask = Variable(input_forced.data.new(*input_forced.size()).bernoulli_(self.force_rate))
                input_t = (input_mask * input_forced) + ((1 - input_mask) * input_gen)
            else:
                if (i == 0):
                    input_t = inputs[i].new_full(inputs[i].size(), 5)
                else:
                    input_t = inputs[i-1]
            # Run a single timestep
            logit, generated, ctx, attn, input_states = self.forward_pass(
                input_t=input_t, keys=keys, values=values, mask=mask, ctx=ctx,
                input_states=input_states
            )
            # Save outputs
            logits.append(logit)
            attns.append(attn)
            generateds.append(generated)

        # For future predictions
        if future > 0:
            assert len(generateds) > 0
            input_t = generateds[-1]
            for _ in range(future):
                # Run a single timestep
                logit, generated, ctx, attn, input_states = self.forward_pass(
                    input_t=input_t, keys=keys, values=values, mask=mask, ctx=ctx,
                    input_states=input_states
                )
                # Save outputs
                logits.append(logit)
                attns.append(attn)
                generateds.append(generated)
                # Pass generated as next input
                input_t = generated

        # Combine all the outputs
        logits = torch.stack(logits, dim=0)  # (L, N, Vocab Size)
        attns = torch.stack(attns, dim=0)  # (L, N, T)
        generateds = torch.stack(generateds, dim=0)  # (L, N)
        return logits, attns, generateds


class Seq2SeqModel(nn.Module):
    # Tie encoder and decoder together
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        self._state_hooks = {}

    def forward(self, utterances, utterance_lengths, chars, char_lengths, future=0):
        keys, values, lengths = self.encoder(utterances, utterance_lengths)
        logits, attns, generated = self.decoder(chars, char_lengths, keys, values, lengths, future=future)
        self._state_hooks['attention'] = attns.permute(1, 0, 2).unsqueeze(1)
        return logits, generated, char_lengths


class SequenceCrossEntropy(nn.CrossEntropyLoss):
    # Customized CrossEntropyLoss
    def __init__(self, *args, **kwargs):
        super(SequenceCrossEntropy, self).__init__(*args, reduce=False, **kwargs)

    def forward(self, prediction, target):
        logits, generated, sequence_lengths = prediction
        maxlen = logits.size(0)
        mask = Variable(output_mask(maxlen, sequence_lengths.data)).float()
        logits = logits * mask.unsqueeze(2)
        losses = super(SequenceCrossEntropy, self).forward(logits.view(-1, logits.size(2)), target.view(-1))
        loss = torch.sum(mask.view(-1) * losses) / logits.size(1)
        return loss
