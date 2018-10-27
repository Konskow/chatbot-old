import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, model_params, embeddings):
        super(Encoder, self).__init__()
        self.n_layers = model_params['encoder_n_layers']
        self.hidden_size = model_params['hidden_size']
        self.embeddings = embeddings

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else model_params['dropout']), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embeddings(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
