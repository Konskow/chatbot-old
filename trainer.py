import os
import random

import torch
from comet_ml import Experiment

from train_utils import batch_to_train_data


def mask_neg_log_loss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    # loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, vocabulary, teacher_forcing_ratio, device='cpu'):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0.0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[vocabulary.word2index['SOS'] for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = mask_neg_log_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = mask_neg_log_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train_iterations(model_name, vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                     embeddings, save_dir,corpus_name, load_filename, model_params, checkpoint, experiment: Experiment, device='cpu'):

    # Load batches for each iteration
    training_batches = [batch_to_train_data(vocabulary, [random.choice(pairs) for _ in range(model_params['batch_size'])])
                        for _ in range(model_params['n_iteration'])]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if load_filename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, model_params['n_iteration'] + 1):
        experiment.log_current_epoch(iteration)

        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embeddings, encoder_optimizer, decoder_optimizer, model_params['batch_size'],
                     model_params['clip'], vocabulary, model_params['teacher_forcing_ratio'], device=device)
        print_loss += loss

        # Print progress
        if iteration % model_params['print_every'] == 0:
            print_loss_avg = print_loss / model_params['print_every']
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
                  .format(iteration, iteration / model_params['n_iteration'] * 100, print_loss_avg))
            experiment.log_metric('loss_avg', print_loss_avg, step=iteration)
            print_loss = 0

        # Save checkpoint
        if iteration % model_params['save_every'] == 0:
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(model_params['encoder_n_layers'],
                                                       model_params['decoder_n_layers'],
                                                       model_params['hidden_size']))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': vocabulary.__dict__,
                'embeddings': embeddings.state_dict(),
                'model_params': model_params,
                'vocabulary': vocabulary
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
