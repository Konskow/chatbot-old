import torch
from torch import nn

from models.attention import Attention


class Decoder(nn.Module):
    def __init__(self, model_params, embeddings, output_size):
        super(Decoder, self).__init__()

        # Keep for reference
        self.attention_mode = model_params['attention_mode']
        self.hidden_size = model_params['hidden_size']
        self.output_size = output_size
        self.n_layers = model_params['decoder_n_layers']
        self.dropout = model_params['dropout']

        # Define layers
        self.embeddings = embeddings
        self.embedding_dropout = nn.Dropout( self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=(0 if self.n_layers == 1 else  self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        self.attention = Attention(self.attention_mode, self.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embeddings(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attention_weights = self.attention(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = torch.nn.functional.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
