from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import random
from io import open

import torch
import torch.nn as nn
from torch import optim
from comet_ml import Experiment

from models.decoder import Decoder
from models.encoder import Encoder
from train_utils import batch_to_train_data
from trainer import train_iterations


def print_lines(file: str, n: int = 10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


if __name__ == "__main__":

    experiment = Experiment(api_key='TL3qwp3y3BmrG2jZ4vsS9nVkv', project_name='chatbot', log_code=False)

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    vocabulary = pickle.load(open('data/voc.pickle', 'rb'))
    pairs = pickle.load(open('data/pairs.pickle', 'rb'))

    # mini_batch_size = 5
    # random_pairs = [random.choice(pairs) for _ in range(mini_batch_size)]
    # batches = batch_to_train_data(vocabulary, random_pairs)
    # input_variable, lengths, target_variable, mask, max_target_len = batches
    #
    # print("input_variable:", input_variable)
    # print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    # print("max_target_len:", max_target_len)

    # Configure models
    model_params = {
        'model_name': 'cb_model',
        'attention_mode': 'dot',
        # attn_model : 'general'
        # attn_model : 'concat'
        'hidden_size': 500,
        'encoder_n_layers': 2,
        'decoder_n_layers': 2,
        'dropout': 0.1,
        'batch_size': 64,
        'teacher_forcing_ratio': 1.0,
        'clip': 50.0,
        'learning_rate': 0.0001,
        'decoder_learning_ratio': 5.0,
        'n_iteration': 4000,
        'print_every': 10,
        'save_every': 500,
        'batch_size': 1
    }

    experiment.log_multiple_params(model_params)

    # Set checkpoint to load from; set to None if starting from scratch
    load_filename = None
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))
    checkpoint = None
    # Load model if a loadFilename is provided
    if load_filename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_filename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocabulary.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embeddings = nn.Embedding(vocabulary.num_words, model_params['hidden_size'])

    if load_filename:
        embeddings.load_state_dict(embedding_sd)

    encoder = Encoder(model_params, embeddings)
    decoder = Decoder(model_params, embeddings, vocabulary.num_words)

    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    model_name = 'chatbot'
    corpus_name = 'corpus'
    save_dir = 'temp2'

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=model_params['learning_rate'])
    decoder_optimizer = optim.Adam(decoder.parameters(),
                                   lr=model_params['learning_rate'] * model_params['decoder_learning_ratio'])
    if load_filename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")

    train_iterations(model_name, vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                     embeddings, save_dir, corpus_name, load_filename, model_params, checkpoint, experiment)
