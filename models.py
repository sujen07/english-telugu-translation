from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden_state=None):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden_state)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, te_vocab):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.vocab = te_vocab

    def forward(self, encoder_outputs, encoder_hidden, max_length=15):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.vocab.lookup_indices(['<bos>'])[0])
        decoder_outputs = []
        decoder_hidden = encoder_hidden

        for _ in range(max_length):
            decoder_out, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_out)
            _, topi  = decoder_out.topk(1)
            decoder_input = topi.squeeze(-1).detach()
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


